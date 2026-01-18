from __future__ import annotations
import abc
import dataclasses
import os
import shlex
import tempfile
import time
import typing as t
from .io import (
from .config import (
from .host_configs import (
from .core_ci import (
from .util import (
from .util_common import (
from .docker_util import (
from .bootstrap import (
from .venv import (
from .ssh import (
from .ansible_util import (
from .containers import (
from .connections import (
from .become import (
from .completion import (
from .dev.container_probe import (
class DockerProfile(ControllerHostProfile[DockerConfig], SshTargetHostProfile[DockerConfig]):
    """Host profile for a docker instance."""
    MARKER = 'ansible-test-marker'

    @dataclasses.dataclass(frozen=True)
    class InitConfig:
        """Configuration details required to run the container init."""
        options: list[str]
        command: str
        command_privileged: bool
        expected_mounts: tuple[CGroupMount, ...]

    @property
    def container_name(self) -> t.Optional[str]:
        """Return the stored container name, if any, otherwise None."""
        return self.state.get('container_name')

    @container_name.setter
    def container_name(self, value: str) -> None:
        """Store the given container name."""
        self.state['container_name'] = value

    @property
    def cgroup_path(self) -> t.Optional[str]:
        """Return the path to the cgroup v1 systemd hierarchy, if any, otherwise None."""
        return self.state.get('cgroup_path')

    @cgroup_path.setter
    def cgroup_path(self, value: str) -> None:
        """Store the path to the cgroup v1 systemd hierarchy."""
        self.state['cgroup_path'] = value

    @property
    def label(self) -> str:
        """Label to apply to resources related to this profile."""
        return f'{('controller' if self.controller else 'target')}'

    def provision(self) -> None:
        """Provision the host before delegation."""
        init_probe = self.args.dev_probe_cgroups is not None
        init_config = self.get_init_config()
        container = run_support_container(args=self.args, context='__test_hosts__', image=self.config.image, name=f'ansible-test-{self.label}', ports=[22], publish_ports=not self.controller, options=init_config.options, cleanup=False, cmd=self.build_init_command(init_config, init_probe))
        if not container:
            if self.args.prime_containers:
                if init_config.command_privileged or init_probe:
                    docker_pull(self.args, UTILITY_IMAGE)
            return
        self.container_name = container.name
        try:
            options = ['--pid', 'host', '--privileged']
            if init_config.command and init_config.command_privileged:
                init_command = init_config.command
                if not init_probe:
                    init_command += f' && {shlex.join(self.wake_command)}'
                cmd = ['nsenter', '-t', str(container.details.container.pid), '-m', '-p', 'sh', '-c', init_command]
                run_utility_container(self.args, f'ansible-test-init-{self.label}', cmd, options)
            if init_probe:
                check_container_cgroup_status(self.args, self.config, self.container_name, init_config.expected_mounts)
                cmd = ['nsenter', '-t', str(container.details.container.pid), '-m', '-p'] + self.wake_command
                run_utility_container(self.args, f'ansible-test-wake-{self.label}', cmd, options)
        except SubprocessError:
            display.info(f'Checking container "{self.container_name}" logs...')
            docker_logs(self.args, self.container_name)
            raise

    def get_init_config(self) -> InitConfig:
        """Return init config for running under the current container engine."""
        self.check_cgroup_requirements()
        engine = require_docker().command
        init_config = getattr(self, f'get_{engine}_init_config')()
        return init_config

    def get_podman_init_config(self) -> InitConfig:
        """Return init config for running under Podman."""
        options = self.get_common_run_options()
        command: t.Optional[str] = None
        command_privileged = False
        expected_mounts: tuple[CGroupMount, ...]
        cgroup_version = get_docker_info(self.args).cgroup_version
        options.extend(('--cap-add', 'SYS_CHROOT'))
        if self.config.audit == AuditMode.REQUIRED and detect_host_properties(self.args).audit_code == 'EPERM':
            options.extend(('--cap-add', 'AUDIT_WRITE'))
        if (loginuid := detect_host_properties(self.args).loginuid) not in (0, LOGINUID_NOT_SET, None):
            display.warning(f'Running containers with capability AUDIT_CONTROL since the container loginuid ({loginuid}) is incorrect. This is most likely due to use of sudo to run ansible-test when loginuid is already set.', unique=True)
            options.extend(('--cap-add', 'AUDIT_CONTROL'))
        if self.config.cgroup == CGroupVersion.NONE:
            options.extend(('--systemd', 'false', '--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
            options.extend(('--systemd', 'always', '--cgroupns', 'host', '--tmpfs', '/sys/fs/cgroup'))
            self.check_systemd_cgroup_v1(options)
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=None, state=CGroupState.HOST), CGroupMount(path=CGroupPath.SYSTEMD_RELEASE_AGENT, type=None, writable=False, state=None))
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
            options.extend(('--systemd', 'always', '--cgroupns', 'private'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
        elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
            cgroup_path = self.create_systemd_cgroup_v1()
            command = f'echo 1 > {cgroup_path}/cgroup.procs'
            options.extend(('--systemd', 'always', '--cgroupns', 'private', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:ro', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=False, state=CGroupState.SHADOWED), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        else:
            raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
        return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)

    def get_docker_init_config(self) -> InitConfig:
        """Return init config for running under Docker."""
        options = self.get_common_run_options()
        command: t.Optional[str] = None
        command_privileged = False
        expected_mounts: tuple[CGroupMount, ...]
        cgroup_version = get_docker_info(self.args).cgroup_version
        if self.config.cgroup == CGroupVersion.NONE:
            if get_docker_info(self.args).cgroupns_option_supported:
                options.extend(('--cgroupns', 'private'))
            options.extend(('--tmpfs', '/sys/fs/cgroup'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None),)
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V1_ONLY) and cgroup_version == 1:
            if get_docker_info(self.args).cgroupns_option_supported:
                options.extend(('--cgroupns', 'host'))
            options.extend(('--tmpfs', '/sys/fs/cgroup', '--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw'))
            self.check_systemd_cgroup_v1(options)
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        elif self.config.cgroup in (CGroupVersion.V1_V2, CGroupVersion.V2_ONLY) and cgroup_version == 2:
            command = 'mount -o remount,rw /sys/fs/cgroup/'
            command_privileged = True
            options.extend(('--cgroupns', 'private'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.CGROUP_V2, writable=True, state=CGroupState.PRIVATE),)
        elif self.config.cgroup == CGroupVersion.V1_ONLY and cgroup_version == 2:
            cgroup_path = self.create_systemd_cgroup_v1()
            command = f'echo 1 > {cgroup_path}/cgroup.procs'
            options.extend(('--cgroupns', 'private', '--tmpfs', '/sys/fs/cgroup', '--tmpfs', '/sys/fs/cgroup/systemd', '--volume', f'{cgroup_path}:{cgroup_path}:rw'))
            expected_mounts = (CGroupMount(path=CGroupPath.ROOT, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=CGroupPath.SYSTEMD, type=MountType.TMPFS, writable=True, state=None), CGroupMount(path=cgroup_path, type=MountType.CGROUP_V1, writable=True, state=CGroupState.HOST))
        else:
            raise InternalError(f'Unhandled cgroup configuration: {self.config.cgroup} on cgroup v{cgroup_version}.')
        return self.InitConfig(options=options, command=command, command_privileged=command_privileged, expected_mounts=expected_mounts)

    def build_init_command(self, init_config: InitConfig, sleep: bool) -> t.Optional[list[str]]:
        """
        Build and return the command to start in the container.
        Returns None if the default command for the container should be used.

        The sleep duration below was selected to:

          - Allow enough time to perform necessary operations in the container before waking it.
          - Make the delay obvious if the wake command doesn't run or succeed.
          - Avoid hanging indefinitely or for an unreasonably long time.

        NOTE: The container must have a POSIX-compliant default shell "sh" with a non-builtin "sleep" command.
              The "sleep" command is invoked through "env" to avoid using a shell builtin "sleep" (if present).
        """
        command = ''
        if init_config.command and (not init_config.command_privileged):
            command += f'{init_config.command} && '
        if sleep or init_config.command_privileged:
            command += 'env sleep 60 ; '
        if not command:
            return None
        docker_pull(self.args, self.config.image)
        inspect = docker_image_inspect(self.args, self.config.image)
        command += f'exec {shlex.join(inspect.cmd)}'
        return ['sh', '-c', command]

    @property
    def wake_command(self) -> list[str]:
        """
        The command used to wake the container from sleep.
        This will be run inside our utility container, so the command used does not need to be present in the container being woken up.
        """
        return ['pkill', 'sleep']

    def check_systemd_cgroup_v1(self, options: list[str]) -> None:
        """Check the cgroup v1 systemd hierarchy to verify it is writeable for our container."""
        probe_script = read_text_file(os.path.join(ANSIBLE_TEST_TARGET_ROOT, 'setup', 'check_systemd_cgroup_v1.sh')).replace('@MARKER@', self.MARKER).replace('@LABEL@', f'{self.label}-{self.args.session_name}')
        cmd = ['sh']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-check-{self.label}', cmd, options, data=probe_script)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                raise ControlGroupError(self.args, f'Unable to create a v1 cgroup within the systemd hierarchy.\nReason: {error}') from ex
            raise

    def create_systemd_cgroup_v1(self) -> str:
        """Create a unique ansible-test cgroup in the v1 systemd hierarchy and return its path."""
        self.cgroup_path = f'/sys/fs/cgroup/systemd/ansible-test-{self.label}-{self.args.session_name}'
        options = ['--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw', '--privileged']
        cmd = ['sh', '-c', f'>&2 echo {shlex.quote(self.MARKER)} && mkdir {shlex.quote(self.cgroup_path)}']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-create-{self.label}', cmd, options)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                raise ControlGroupError(self.args, f'Unable to create a v1 cgroup within the systemd hierarchy.\nReason: {error}') from ex
            raise
        return self.cgroup_path

    @property
    def delete_systemd_cgroup_v1_command(self) -> list[str]:
        """The command used to remove the previously created ansible-test cgroup in the v1 systemd hierarchy."""
        return ['find', self.cgroup_path, '-type', 'd', '-delete']

    def delete_systemd_cgroup_v1(self) -> None:
        """Delete a previously created ansible-test cgroup in the v1 systemd hierarchy."""
        options = ['--volume', '/sys/fs/cgroup/systemd:/sys/fs/cgroup/systemd:rw', '--privileged']
        cmd = ['sh', '-c', f'>&2 echo {shlex.quote(self.MARKER)} && {shlex.join(self.delete_systemd_cgroup_v1_command)}']
        try:
            run_utility_container(self.args, f'ansible-test-cgroup-delete-{self.label}', cmd, options)
        except SubprocessError as ex:
            if (error := self.extract_error(ex.stderr)):
                if error.endswith(': No such file or directory'):
                    return
            display.error(str(ex))

    def extract_error(self, value: str) -> t.Optional[str]:
        """
        Extract the ansible-test portion of the error message from the given value and return it.
        Returns None if no ansible-test marker was found.
        """
        lines = value.strip().splitlines()
        try:
            idx = lines.index(self.MARKER)
        except ValueError:
            return None
        lines = lines[idx + 1:]
        message = '\n'.join(lines)
        return message

    def check_cgroup_requirements(self) -> None:
        """Check cgroup requirements for the container."""
        cgroup_version = get_docker_info(self.args).cgroup_version
        if cgroup_version not in (1, 2):
            raise ApplicationError(f'The container host provides cgroup v{cgroup_version}, but only version v1 and v2 are supported.')
        if self.config.cgroup == CGroupVersion.V2_ONLY and cgroup_version != 2:
            raise ApplicationError(f'Container {self.config.name} requires cgroup v2 but the container host provides cgroup v{cgroup_version}.')
        if self.config.cgroup == CGroupVersion.V1_ONLY or (self.config.cgroup != CGroupVersion.NONE and get_docker_info(self.args).cgroup_version == 1):
            if (cgroup_v1 := detect_host_properties(self.args).cgroup_v1) != SystemdControlGroupV1Status.VALID:
                if self.config.cgroup == CGroupVersion.V1_ONLY:
                    if get_docker_info(self.args).cgroup_version == 2:
                        reason = f'Container {self.config.name} requires cgroup v1, but the container host only provides cgroup v2.'
                    else:
                        reason = f'Container {self.config.name} requires cgroup v1, but the container host does not appear to be running systemd.'
                else:
                    reason = 'The container host provides cgroup v1, but does not appear to be running systemd.'
                reason += f'\n{cgroup_v1.value}'
                raise ControlGroupError(self.args, reason)

    def setup(self) -> None:
        """Perform out-of-band setup before delegation."""
        bootstrapper = BootstrapDocker(controller=self.controller, python_versions=[self.python.version], ssh_key=SshKey(self.args))
        setup_sh = bootstrapper.get_script()
        shell = setup_sh.splitlines()[0][2:]
        try:
            docker_exec(self.args, self.container_name, [shell], data=setup_sh, capture=False)
        except SubprocessError:
            display.info(f'Checking container "{self.container_name}" logs...')
            docker_logs(self.args, self.container_name)
            raise

    def deprovision(self) -> None:
        """Deprovision the host after delegation has completed."""
        container_exists = False
        if self.container_name:
            if self.args.docker_terminate == TerminateMode.ALWAYS or (self.args.docker_terminate == TerminateMode.SUCCESS and self.args.success):
                docker_rm(self.args, self.container_name)
            else:
                container_exists = True
        if self.cgroup_path:
            if container_exists:
                display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing. Then run `{shlex.join(self.delete_systemd_cgroup_v1_command)}` on the container host.')
            else:
                self.delete_systemd_cgroup_v1()
        elif container_exists:
            display.notice(f'Remember to run `{require_docker().command} rm -f {self.container_name}` when finished testing.')

    def wait(self) -> None:
        """Wait for the instance to be ready. Executed before delegation for the controller and after delegation for targets."""
        if not self.controller:
            con = self.get_controller_target_connections()[0]
            last_error = ''
            for dummy in range(1, 10):
                try:
                    con.run(['id'], capture=True)
                except SubprocessError as ex:
                    if 'Permission denied' in ex.message:
                        raise
                    last_error = str(ex)
                    time.sleep(1)
                else:
                    return
            display.info('Checking SSH debug output...')
            display.info(last_error)
            if not self.args.delegate and (not self.args.host_path):

                def callback() -> None:
                    """Callback to run during error display."""
                    self.on_target_failure()
            else:
                callback = None
            raise HostConnectionError(f'Timeout waiting for {self.config.name} container {self.container_name}.', callback)

    def get_controller_target_connections(self) -> list[SshConnection]:
        """Return SSH connection(s) for accessing the host as a target from the controller."""
        containers = get_container_database(self.args)
        access = containers.data[HostType.control]['__test_hosts__'][self.container_name]
        host = access.host_ip
        port = dict(access.port_map())[22]
        settings = SshConnectionDetail(name=self.config.name, user='root', host=host, port=port, identity_file=SshKey(self.args).key, python_interpreter=self.python.path, enable_rsa_sha1='centos6' in self.config.image)
        return [SshConnection(self.args, settings)]

    def get_origin_controller_connection(self) -> DockerConnection:
        """Return a connection for accessing the host as a controller from the origin."""
        return DockerConnection(self.args, self.container_name)

    def get_working_directory(self) -> str:
        """Return the working directory for the host."""
        return '/root'

    def on_target_failure(self) -> None:
        """Executed during failure handling if this profile is a target."""
        display.info(f'Checking container "{self.container_name}" logs...')
        try:
            docker_logs(self.args, self.container_name)
        except SubprocessError as ex:
            display.error(str(ex))
        if self.config.cgroup != CGroupVersion.NONE:
            display.info(f'Checking container "{self.container_name}" systemd logs...')
            try:
                docker_exec(self.args, self.container_name, ['journalctl'], capture=False)
            except SubprocessError as ex:
                display.error(str(ex))
        display.error(f'Connection to container "{self.container_name}" failed. See logs and original error above.')

    def get_common_run_options(self) -> list[str]:
        """Return a list of options needed to run the container."""
        options = ['--tmpfs', '/tmp:exec', '--tmpfs', '/run:exec', '--tmpfs', '/run/lock']
        if self.config.privileged:
            options.append('--privileged')
        if self.config.memory:
            options.extend([f'--memory={self.config.memory}', f'--memory-swap={self.config.memory}'])
        if self.config.seccomp != 'default':
            options.extend(['--security-opt', f'seccomp={self.config.seccomp}'])
        docker_socket = '/var/run/docker.sock'
        if get_docker_hostname() != 'localhost' or os.path.exists(docker_socket):
            options.extend(['--volume', f'{docker_socket}:{docker_socket}'])
        return options