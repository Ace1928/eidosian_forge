import os
import signal
import sys
from functools import wraps
import click
from kombu.utils.objects import cached_property
from celery import VERSION_BANNER
from celery.apps.multi import Cluster, MultiParser, NamespacedOptionParser
from celery.bin.base import CeleryCommand, handle_preload_options
from celery.platforms import EX_FAILURE, EX_OK, signals
from celery.utils import term
from celery.utils.text import pluralize
class MultiTool(TermLogger):
    """The ``celery multi`` program."""
    MultiParser = MultiParser
    OptionParser = NamespacedOptionParser
    reserved_options = [('--nosplash', 'nosplash'), ('--quiet', 'quiet'), ('-q', 'quiet'), ('--verbose', 'verbose'), ('--no-color', 'no_color')]

    def __init__(self, env=None, cmd=None, fh=None, stdout=None, stderr=None, **kwargs):
        self.env = env
        self.cmd = cmd
        self.setup_terminal(stdout or fh, stderr, **kwargs)
        self.fh = self.stdout
        self.prog_name = 'celery multi'
        self.commands = {'start': self.start, 'show': self.show, 'stop': self.stop, 'stopwait': self.stopwait, 'stop_verify': self.stopwait, 'restart': self.restart, 'kill': self.kill, 'names': self.names, 'expand': self.expand, 'get': self.get, 'help': self.help}

    def execute_from_commandline(self, argv, cmd=None):
        argv = self._handle_reserved_options(argv)
        self.cmd = cmd if cmd is not None else self.cmd
        self.prog_name = os.path.basename(argv.pop(0))
        if not self.validate_arguments(argv):
            return self.error()
        return self.call_command(argv[0], argv[1:])

    def validate_arguments(self, argv):
        return argv and argv[0][0] != '-'

    def call_command(self, command, argv):
        try:
            return self.commands[command](*argv) or EX_OK
        except KeyError:
            return self.error(f'Invalid command: {command}')

    def _handle_reserved_options(self, argv):
        argv = list(argv)
        for arg, attr in self.reserved_options:
            if arg in argv:
                setattr(self, attr, bool(argv.pop(argv.index(arg))))
        return argv

    @splash
    @using_cluster
    def start(self, cluster):
        self.note('> Starting nodes...')
        return int(any(cluster.start()))

    @splash
    @using_cluster_and_sig
    def stop(self, cluster, sig, **kwargs):
        return cluster.stop(sig=sig, **kwargs)

    @splash
    @using_cluster_and_sig
    def stopwait(self, cluster, sig, **kwargs):
        return cluster.stopwait(sig=sig, **kwargs)
    stop_verify = stopwait

    @splash
    @using_cluster_and_sig
    def restart(self, cluster, sig, **kwargs):
        return int(any(cluster.restart(sig=sig, **kwargs)))

    @using_cluster
    def names(self, cluster):
        self.say('\n'.join((n.name for n in cluster)))

    def get(self, wanted, *argv):
        try:
            node = self.cluster_from_argv(argv).find(wanted)
        except KeyError:
            return EX_FAILURE
        else:
            return self.ok(' '.join(node.argv))

    @using_cluster
    def show(self, cluster):
        return self.ok('\n'.join((' '.join(node.argv_with_executable) for node in cluster)))

    @splash
    @using_cluster
    def kill(self, cluster):
        return cluster.kill()

    def expand(self, template, *argv):
        return self.ok('\n'.join((node.expander(template) for node in self.cluster_from_argv(argv))))

    def help(self, *argv):
        self.say(__doc__)

    def _find_sig_argument(self, p, default=signal.SIGTERM):
        args = p.args[len(p.values):]
        for arg in reversed(args):
            if len(arg) == 2 and arg[0] == '-':
                try:
                    return int(arg[1])
                except ValueError:
                    pass
            if arg[0] == '-':
                try:
                    return signals.signum(arg[1:])
                except (AttributeError, TypeError):
                    pass
        return default

    def _nodes_from_argv(self, argv, cmd=None):
        cmd = cmd if cmd is not None else self.cmd
        p = self.OptionParser(argv)
        p.parse()
        return (p, self.MultiParser(cmd=cmd).parse(p))

    def cluster_from_argv(self, argv, cmd=None):
        _, cluster = self._cluster_from_argv(argv, cmd=cmd)
        return cluster

    def _cluster_from_argv(self, argv, cmd=None):
        p, nodes = self._nodes_from_argv(argv, cmd=cmd)
        return (p, self.Cluster(list(nodes), cmd=cmd))

    def Cluster(self, nodes, cmd=None):
        return Cluster(nodes, cmd=cmd, env=self.env, on_stopping_preamble=self.on_stopping_preamble, on_send_signal=self.on_send_signal, on_still_waiting_for=self.on_still_waiting_for, on_still_waiting_progress=self.on_still_waiting_progress, on_still_waiting_end=self.on_still_waiting_end, on_node_start=self.on_node_start, on_node_restart=self.on_node_restart, on_node_shutdown_ok=self.on_node_shutdown_ok, on_node_status=self.on_node_status, on_node_signal_dead=self.on_node_signal_dead, on_node_signal=self.on_node_signal, on_node_down=self.on_node_down, on_child_spawn=self.on_child_spawn, on_child_signalled=self.on_child_signalled, on_child_failure=self.on_child_failure)

    def on_stopping_preamble(self, nodes):
        self.note(self.colored.blue('> Stopping nodes...'))

    def on_send_signal(self, node, sig):
        self.note('\t> {0.name}: {1} -> {0.pid}'.format(node, sig))

    def on_still_waiting_for(self, nodes):
        num_left = len(nodes)
        if num_left:
            self.note(self.colored.blue('> Waiting for {} {} -> {}...'.format(num_left, pluralize(num_left, 'node'), ', '.join((str(node.pid) for node in nodes)))), newline=False)

    def on_still_waiting_progress(self, nodes):
        self.note('.', newline=False)

    def on_still_waiting_end(self):
        self.note('')

    def on_node_signal_dead(self, node):
        self.note('Could not signal {0.name} ({0.pid}): No such process'.format(node))

    def on_node_start(self, node):
        self.note(f'\t> {node.name}: ', newline=False)

    def on_node_restart(self, node):
        self.note(self.colored.blue(f'> Restarting node {node.name}: '), newline=False)

    def on_node_down(self, node):
        self.note(f'> {node.name}: {self.DOWN}')

    def on_node_shutdown_ok(self, node):
        self.note(f'\n\t> {node.name}: {self.OK}')

    def on_node_status(self, node, retval):
        self.note(retval and self.FAILED or self.OK)

    def on_node_signal(self, node, sig):
        self.note('Sending {sig} to node {0.name} ({0.pid})'.format(node, sig=sig))

    def on_child_spawn(self, node, argstr, env):
        self.info(f'  {argstr}')

    def on_child_signalled(self, node, signum):
        self.note(f'* Child was terminated by signal {signum}')

    def on_child_failure(self, node, retcode):
        self.note(f'* Child terminated with exit code {retcode}')

    @cached_property
    def OK(self):
        return str(self.colored.green('OK'))

    @cached_property
    def FAILED(self):
        return str(self.colored.red('FAILED'))

    @cached_property
    def DOWN(self):
        return str(self.colored.magenta('DOWN'))