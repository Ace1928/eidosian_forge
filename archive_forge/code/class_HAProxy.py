from __future__ import absolute_import, division, print_function
import csv
import socket
import time
from string import Template
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_bytes, to_text
class HAProxy(object):
    """
    Used for communicating with HAProxy through its local UNIX socket interface.
    Perform common tasks in Haproxy related to enable server and
    disable server.

    The complete set of external commands Haproxy handles is documented
    on their website:

    http://haproxy.1wt.eu/download/1.5/doc/configuration.txt#Unix Socket commands
    """

    def __init__(self, module):
        self.module = module
        self.state = self.module.params['state']
        self.host = self.module.params['host']
        self.backend = self.module.params['backend']
        self.weight = self.module.params['weight']
        self.socket = self.module.params['socket']
        self.shutdown_sessions = self.module.params['shutdown_sessions']
        self.fail_on_not_found = self.module.params['fail_on_not_found']
        self.agent = self.module.params['agent']
        self.health = self.module.params['health']
        self.wait = self.module.params['wait']
        self.wait_retries = self.module.params['wait_retries']
        self.wait_interval = self.module.params['wait_interval']
        self._drain = self.module.params['drain']
        self.command_results = {}

    def execute(self, cmd, timeout=200, capture_output=True):
        """
        Executes a HAProxy command by sending a message to a HAProxy's local
        UNIX socket and waiting up to 'timeout' milliseconds for the response.
        """
        self.client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.client.connect(self.socket)
        self.client.sendall(to_bytes('%s\n' % cmd))
        result = b''
        buf = b''
        buf = self.client.recv(RECV_SIZE)
        while buf:
            result += buf
            buf = self.client.recv(RECV_SIZE)
        result = to_text(result, errors='surrogate_or_strict')
        if capture_output:
            self.capture_command_output(cmd, result.strip())
        self.client.close()
        return result

    def capture_command_output(self, cmd, output):
        """
        Capture the output for a command
        """
        if 'command' not in self.command_results:
            self.command_results['command'] = []
        self.command_results['command'].append(cmd)
        if 'output' not in self.command_results:
            self.command_results['output'] = []
        self.command_results['output'].append(output)

    def discover_all_backends(self):
        """
        Discover all entries with svname = 'BACKEND' and return a list of their corresponding
        pxnames
        """
        data = self.execute('show stat', 200, False).lstrip('# ')
        r = csv.DictReader(data.splitlines())
        return tuple(map(lambda d: d['pxname'], filter(lambda d: d['svname'] == 'BACKEND', r)))

    def discover_version(self):
        """
        Attempt to extract the haproxy version.
        Return a tuple containing major and minor version.
        """
        data = self.execute('show info', 200, False)
        lines = data.splitlines()
        line = [x for x in lines if 'Version:' in x]
        try:
            version_values = line[0].partition(':')[2].strip().split('.', 3)
            version = (int(version_values[0]), int(version_values[1]))
        except (ValueError, TypeError, IndexError):
            version = None
        return version

    def execute_for_backends(self, cmd, pxname, svname, wait_for_status=None):
        """
        Run some command on the specified backends. If no backends are provided they will
        be discovered automatically (all backends)
        """
        if pxname is None:
            backends = self.discover_all_backends()
        else:
            backends = [pxname]
        for backend in backends:
            state = self.get_state_for(backend, svname)
            if self.fail_on_not_found and state is None:
                self.module.fail_json(msg="The specified backend '%s/%s' was not found!" % (backend, svname))
            if state is not None:
                self.execute(Template(cmd).substitute(pxname=backend, svname=svname))
                if self.wait:
                    self.wait_until_status(backend, svname, wait_for_status)

    def get_state_for(self, pxname, svname):
        """
        Find the state of specific services. When pxname is not set, get all backends for a specific host.
        Returns a list of dictionaries containing the status and weight for those services.
        """
        data = self.execute('show stat', 200, False).lstrip('# ')
        r = csv.DictReader(data.splitlines())
        state = tuple(map(lambda d: {'status': d['status'], 'weight': d['weight'], 'scur': d['scur']}, filter(lambda d: (pxname is None or d['pxname'] == pxname) and d['svname'] == svname, r)))
        return state or None

    def wait_until_status(self, pxname, svname, status):
        """
        Wait for a service to reach the specified status. Try RETRIES times
        with INTERVAL seconds of sleep in between. If the service has not reached
        the expected status in that time, the module will fail. If the service was
        not found, the module will fail.
        """
        for i in range(1, self.wait_retries):
            state = self.get_state_for(pxname, svname)
            if status in state[0]['status']:
                if not self._drain or state[0]['scur'] == '0':
                    return True
            time.sleep(self.wait_interval)
        self.module.fail_json(msg="server %s/%s not status '%s' after %d retries. Aborting." % (pxname, svname, status, self.wait_retries))

    def enabled(self, host, backend, weight):
        """
        Enabled action, marks server to UP and checks are re-enabled,
        also supports to get current weight for server (default) and
        set the weight for haproxy backend server when provides.
        """
        cmd = 'get weight $pxname/$svname; enable server $pxname/$svname'
        if self.agent:
            cmd += '; enable agent $pxname/$svname'
        if self.health:
            cmd += '; enable health $pxname/$svname'
        if weight:
            cmd += '; set weight $pxname/$svname %s' % weight
        self.execute_for_backends(cmd, backend, host, 'UP')

    def disabled(self, host, backend, shutdown_sessions):
        """
        Disabled action, marks server to DOWN for maintenance. In this mode, no more checks will be
        performed on the server until it leaves maintenance,
        also it shutdown sessions while disabling backend host server.
        """
        cmd = 'get weight $pxname/$svname'
        if self.agent:
            cmd += '; disable agent $pxname/$svname'
        if self.health:
            cmd += '; disable health $pxname/$svname'
        cmd += '; disable server $pxname/$svname'
        if shutdown_sessions:
            cmd += '; shutdown sessions server $pxname/$svname'
        self.execute_for_backends(cmd, backend, host, 'MAINT')

    def drain(self, host, backend, status='DRAIN'):
        """
        Drain action, sets the server to DRAIN mode.
        In this mode, the server will not accept any new connections
        other than those that are accepted via persistence.
        """
        haproxy_version = self.discover_version()
        if haproxy_version and (1, 5) <= haproxy_version:
            cmd = 'set server $pxname/$svname state drain'
            self.execute_for_backends(cmd, backend, host, 'DRAIN')
            if status == 'MAINT':
                self.disabled(host, backend, self.shutdown_sessions)

    def act(self):
        """
        Figure out what you want to do from ansible, and then do it.
        """
        self.command_results['state_before'] = self.get_state_for(self.backend, self.host)
        if self.state == 'enabled':
            self.enabled(self.host, self.backend, self.weight)
        elif self.state == 'disabled' and self._drain:
            self.drain(self.host, self.backend, status='MAINT')
        elif self.state == 'disabled':
            self.disabled(self.host, self.backend, self.shutdown_sessions)
        elif self.state == 'drain':
            self.drain(self.host, self.backend)
        else:
            self.module.fail_json(msg="unknown state specified: '%s'" % self.state)
        self.command_results['state_after'] = self.get_state_for(self.backend, self.host)
        self.command_results['changed'] = self.command_results['state_before'] != self.command_results['state_after']
        self.module.exit_json(**self.command_results)