from .. import errors
from ..utils.utils import (
from .base import DictType
from .healthcheck import Healthcheck
class ContainerConfig(dict):

    def __init__(self, version, image, command, hostname=None, user=None, detach=False, stdin_open=False, tty=False, ports=None, environment=None, volumes=None, network_disabled=False, entrypoint=None, working_dir=None, domainname=None, host_config=None, mac_address=None, labels=None, stop_signal=None, networking_config=None, healthcheck=None, stop_timeout=None, runtime=None):
        if stop_timeout is not None and version_lt(version, '1.25'):
            raise errors.InvalidVersion('stop_timeout was only introduced in API version 1.25')
        if healthcheck is not None:
            if version_lt(version, '1.24'):
                raise errors.InvalidVersion('Health options were only introduced in API version 1.24')
            if version_lt(version, '1.29') and 'StartPeriod' in healthcheck:
                raise errors.InvalidVersion('healthcheck start period was introduced in API version 1.29')
        if isinstance(command, str):
            command = split_command(command)
        if isinstance(entrypoint, str):
            entrypoint = split_command(entrypoint)
        if isinstance(environment, dict):
            environment = format_environment(environment)
        if isinstance(labels, list):
            labels = {lbl: '' for lbl in labels}
        if isinstance(ports, list):
            exposed_ports = {}
            for port_definition in ports:
                port = port_definition
                proto = 'tcp'
                if isinstance(port_definition, tuple):
                    if len(port_definition) == 2:
                        proto = port_definition[1]
                    port = port_definition[0]
                exposed_ports[f'{port}/{proto}'] = {}
            ports = exposed_ports
        if isinstance(volumes, str):
            volumes = [volumes]
        if isinstance(volumes, list):
            volumes_dict = {}
            for vol in volumes:
                volumes_dict[vol] = {}
            volumes = volumes_dict
        if healthcheck and isinstance(healthcheck, dict):
            healthcheck = Healthcheck(**healthcheck)
        attach_stdin = False
        attach_stdout = False
        attach_stderr = False
        stdin_once = False
        if not detach:
            attach_stdout = True
            attach_stderr = True
            if stdin_open:
                attach_stdin = True
                stdin_once = True
        self.update({'Hostname': hostname, 'Domainname': domainname, 'ExposedPorts': ports, 'User': str(user) if user is not None else None, 'Tty': tty, 'OpenStdin': stdin_open, 'StdinOnce': stdin_once, 'AttachStdin': attach_stdin, 'AttachStdout': attach_stdout, 'AttachStderr': attach_stderr, 'Env': environment, 'Cmd': command, 'Image': image, 'Volumes': volumes, 'NetworkDisabled': network_disabled, 'Entrypoint': entrypoint, 'WorkingDir': working_dir, 'HostConfig': host_config, 'NetworkingConfig': networking_config, 'MacAddress': mac_address, 'Labels': labels, 'StopSignal': stop_signal, 'Healthcheck': healthcheck, 'StopTimeout': stop_timeout, 'Runtime': runtime})