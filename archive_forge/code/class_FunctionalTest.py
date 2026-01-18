import abc
import atexit
import datetime
import errno
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from testtools import content as ttc
import textwrap
import time
from unittest import mock
import urllib.parse as urlparse
import uuid
import fixtures
import glance_store
from os_win import utilsfactory as os_win_utilsfactory
from oslo_config import cfg
from oslo_serialization import jsonutils
import testtools
import webob
from glance.common import config
from glance.common import utils
from glance.common import wsgi
from glance.db.sqlalchemy import api as db_api
from glance import tests as glance_tests
from glance.tests import utils as test_utils
import glance.async_
class FunctionalTest(test_utils.BaseTestCase):
    """
    Base test class for any test that wants to test the actual
    servers and clients and not just the stubbed out interfaces
    """
    inited = False
    disabled = False
    launched_servers = []

    def setUp(self):
        super(FunctionalTest, self).setUp()
        self.test_dir = self.useFixture(fixtures.TempDir()).path
        self.api_protocol = 'http'
        self.api_port, api_sock = test_utils.get_unused_port_and_socket()
        self.include_scrubber = True
        self.config(bind_host='127.0.0.1')
        self.config(image_cache_dir=self.test_dir)
        self.tracecmd = tracecmd_osmap.get(platform.system())
        conf_dir = os.path.join(self.test_dir, 'etc')
        utils.safe_mkdirs(conf_dir)
        self.copy_data_file('schema-image.json', conf_dir)
        self.copy_data_file('property-protections.conf', conf_dir)
        self.copy_data_file('property-protections-policies.conf', conf_dir)
        self.property_file_roles = os.path.join(conf_dir, 'property-protections.conf')
        property_policies = 'property-protections-policies.conf'
        self.property_file_policies = os.path.join(conf_dir, property_policies)
        self.policy_file = os.path.join(conf_dir, 'policy.yaml')
        self.api_server = ApiServer(self.test_dir, self.api_port, self.policy_file, sock=api_sock)
        self.scrubber_daemon = ScrubberDaemon(self.test_dir, self.policy_file)
        self.pid_files = [self.api_server.pid_file, self.scrubber_daemon.pid_file]
        self.files_to_destroy = []
        self.launched_servers = []
        self._attached_server_logs = []
        self.addOnException(self.add_log_details_on_exception)
        if not self.disabled:
            self.addCleanup(self._reset_database, self.api_server.sql_connection)
            self.addCleanup(self.cleanup)
            self._reset_database(self.api_server.sql_connection)

    def _url(self, path):
        return 'http://127.0.0.1:%d%s' % (self.api_port, path)

    def set_policy_rules(self, rules):
        fap = open(self.policy_file, 'w')
        fap.write(jsonutils.dumps(rules))
        fap.close()

    def _reset_database(self, conn_string):
        conn_pieces = urlparse.urlparse(conn_string)
        if conn_string.startswith('sqlite'):
            pass
        elif conn_string.startswith('mysql'):
            database = conn_pieces.path.strip('/')
            loc_pieces = conn_pieces.netloc.split('@')
            host = loc_pieces[1]
            auth_pieces = loc_pieces[0].split(':')
            user = auth_pieces[0]
            password = ''
            if len(auth_pieces) > 1:
                if auth_pieces[1].strip():
                    password = '-p%s' % auth_pieces[1]
            sql = 'drop database if exists %(database)s; create database %(database)s;' % {'database': database}
            cmd = 'mysql -u%(user)s %(password)s -h%(host)s -e"%(sql)s"' % {'user': user, 'password': password, 'host': host, 'sql': sql}
            exitcode, out, err = execute(cmd)
            self.assertEqual(0, exitcode)

    def cleanup(self):
        """
        Makes sure anything we created or started up in the
        tests are destroyed or spun down
        """
        servers = [self.api_server, self.scrubber_daemon]
        for s in servers:
            try:
                s.stop()
            except Exception:
                pass
        for f in self.files_to_destroy:
            if os.path.exists(f):
                os.unlink(f)

    def start_server(self, server, expect_launch, expect_exit=True, expected_exitcode=0, **kwargs):
        """
        Starts a server on an unused port.

        Any kwargs passed to this method will override the configuration
        value in the conf file used in starting the server.

        :param server: the server to launch
        :param expect_launch: true iff the server is expected to
                              successfully start
        :param expect_exit: true iff the launched process is expected
                            to exit in a timely fashion
        :param expected_exitcode: expected exitcode from the launcher
        """
        self.cleanup()
        exitcode, out, err = server.start(expect_exit=expect_exit, expected_exitcode=expected_exitcode, **kwargs)
        if expect_exit:
            self.assertEqual(expected_exitcode, exitcode, 'Failed to spin up the requested server. Got: %s' % err)
        self.launched_servers.append(server)
        launch_msg = self.wait_for_servers([server], expect_launch)
        self.assertTrue(launch_msg is None, launch_msg)

    def start_with_retry(self, server, port_name, max_retries, expect_launch=True, **kwargs):
        """
        Starts a server, with retries if the server launches but
        fails to start listening on the expected port.

        :param server: the server to launch
        :param port_name: the name of the port attribute
        :param max_retries: the maximum number of attempts
        :param expect_launch: true iff the server is expected to
                              successfully start
        :param expect_exit: true iff the launched process is expected
                            to exit in a timely fashion
        """
        launch_msg = None
        for i in range(max_retries):
            exitcode, out, err = server.start(expect_exit=not expect_launch, **kwargs)
            name = server.server_name
            self.assertEqual(0, exitcode, 'Failed to spin up the %s server. Got: %s' % (name, err))
            launch_msg = self.wait_for_servers([server], expect_launch)
            if launch_msg:
                server.stop()
                server.bind_port = get_unused_port()
                setattr(self, port_name, server.bind_port)
            else:
                self.launched_servers.append(server)
                break
        self.assertTrue(launch_msg is None, launch_msg)

    def start_servers(self, **kwargs):
        """
        Starts the API and Registry servers (glance-control api start
        ) on unused ports.  glance-control
        should be installed into the python path

        Any kwargs passed to this method will override the configuration
        value in the conf file used in starting the servers.
        """
        self.cleanup()
        self.start_with_retry(self.api_server, 'api_port', 3, **kwargs)
        if self.include_scrubber:
            exitcode, out, err = self.scrubber_daemon.start(**kwargs)
            self.assertEqual(0, exitcode, 'Failed to spin up the Scrubber daemon. Got: %s' % err)

    def ping_server(self, port):
        """
        Simple ping on the port. If responsive, return True, else
        return False.

        :note We use raw sockets, not ping here, since ping uses ICMP and
        has no concept of ports...
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(('127.0.0.1', port))
            return True
        except socket.error:
            return False
        finally:
            s.close()

    def ping_server_ipv6(self, port):
        """
        Simple ping on the port. If responsive, return True, else
        return False.

        :note We use raw sockets, not ping here, since ping uses ICMP and
        has no concept of ports...

        The function uses IPv6 (therefore AF_INET6 and ::1).
        """
        s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        try:
            s.connect(('::1', port))
            return True
        except socket.error:
            return False
        finally:
            s.close()

    def wait_for_servers(self, servers, expect_launch=True, timeout=30):
        """
        Tight loop, waiting for the given server port(s) to be available.
        Returns when all are pingable. There is a timeout on waiting
        for the servers to come up.

        :param servers: Glance server ports to ping
        :param expect_launch: Optional, true iff the server(s) are
                              expected to successfully start
        :param timeout: Optional, defaults to 30 seconds
        :returns: None if launch expectation is met, otherwise an
                 assertion message
        """
        now = datetime.datetime.now()
        timeout_time = now + datetime.timedelta(seconds=timeout)
        replied = []
        while timeout_time > now:
            pinged = 0
            for server in servers:
                if self.ping_server(server.bind_port):
                    pinged += 1
                    if server not in replied:
                        replied.append(server)
            if pinged == len(servers):
                msg = 'Unexpected server launch status'
                return None if expect_launch else msg
            now = datetime.datetime.now()
            time.sleep(0.05)
        failed = list(set(servers) - set(replied))
        msg = 'Unexpected server launch status for: '
        for f in failed:
            msg += '%s, ' % f.server_name
            if os.path.exists(f.pid_file):
                pid = f.process_pid
                trace = f.pid_file.replace('.pid', '.trace')
                if self.tracecmd:
                    cmd = '%s -p %d -o %s' % (self.tracecmd, pid, trace)
                    try:
                        execute(cmd, raise_error=False, expect_exit=False)
                    except OSError as e:
                        if e.errno == errno.ENOENT:
                            raise RuntimeError('No executable found for "%s" command.' % self.tracecmd)
                        else:
                            raise
                    time.sleep(0.5)
                    if os.path.exists(trace):
                        msg += '\n%s:\n%s\n' % (self.tracecmd, open(trace).read())
        self.add_log_details(failed)
        return msg if expect_launch else None

    def stop_server(self, server):
        """
        Called to stop a single server in a normal fashion using the
        glance-control stop method to gracefully shut the server down.

        :param server: the server to stop
        """
        server.stop()

    def stop_servers(self):
        """
        Called to stop the started servers in a normal fashion. Note
        that cleanup() will stop the servers using a fairly draconian
        method of sending a SIGTERM signal to the servers. Here, we use
        the glance-control stop method to gracefully shut the server down.
        This method also asserts that the shutdown was clean, and so it
        is meant to be called during a normal test case sequence.
        """
        self.stop_server(self.api_server)
        if self.include_scrubber:
            self.stop_server(self.scrubber_daemon)

    def run_sql_cmd(self, sql):
        """
        Provides a crude mechanism to run manual SQL commands for backend
        DB verification within the functional tests.
        The raw result set is returned.
        """
        engine = db_api.get_engine()
        return engine.execute(sql)

    def copy_data_file(self, file_name, dst_dir):
        src_file_name = os.path.join('glance/tests/etc', file_name)
        shutil.copy(src_file_name, dst_dir)
        dst_file_name = os.path.join(dst_dir, file_name)
        return dst_file_name

    def add_log_details_on_exception(self, *args, **kwargs):
        self.add_log_details()

    def add_log_details(self, servers=None):
        for s in servers or self.launched_servers:
            if s.log_file not in self._attached_server_logs:
                self._attached_server_logs.append(s.log_file)
            self.addDetail(s.server_name, testtools.content.text_content(s.dump_log()))