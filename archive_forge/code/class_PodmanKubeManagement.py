from __future__ import absolute_import, division, print_function
import re  # noqa: F402
from ansible.module_utils.basic import AnsibleModule  # noqa: F402
class PodmanKubeManagement:

    def __init__(self, module, executable):
        self.module = module
        self.actions = []
        self.executable = executable
        self.command = [self.executable, 'play', 'kube']
        creds = []
        if self.module.params['annotation']:
            for k, v in self.module.params['annotation'].items():
                self.command.extend(['--annotation', '"{k}={v}"'.format(k=k, v=v)])
        if self.module.params['username']:
            creds += [self.module.params['username']]
            if self.module.params['password']:
                creds += [self.module.params['password']]
            creds = ':'.join(creds)
            self.command.extend(['--creds=%s' % creds])
        if self.module.params['network']:
            networks = ','.join(self.module.params['network'])
            self.command.extend(['--network=%s' % networks])
        if self.module.params['configmap']:
            configmaps = ','.join(self.module.params['configmap'])
            self.command.extend(['--configmap=%s' % configmaps])
        if self.module.params['log_opt']:
            for k, v in self.module.params['log_opt'].items():
                self.command.extend(['--log-opt', '{k}={v}'.format(k=k.replace('_', '-'), v=v)])
        start = self.module.params['state'] == 'started'
        self.command.extend(['--start=%s' % str(start).lower()])
        for arg, param in {'--authfile': 'authfile', '--build': 'build', '--cert-dir': 'cert_dir', '--context-dir': 'context_dir', '--log-driver': 'log_driver', '--seccomp-profile-root': 'seccomp_profile_root', '--tls-verify': 'tls_verify', '--log-level': 'log_level', '--userns': 'userns', '--quiet': 'quiet'}.items():
            if self.module.params[param] is not None:
                self.command += ['%s=%s' % (arg, self.module.params[param])]
        self.command += [self.module.params['kube_file']]

    def _command_run(self, cmd):
        rc, out, err = self.module.run_command(cmd)
        self.actions.append(' '.join(cmd))
        if self.module.params['debug']:
            self.module.log('PODMAN-PLAY-KUBE command: %s' % ' '.join(cmd))
            self.module.log('PODMAN-PLAY-KUBE stdout: %s' % out)
            self.module.log('PODMAN-PLAY-KUBE stderr: %s' % err)
            self.module.log('PODMAN-PLAY-KUBE rc: %s' % rc)
        return (rc, out, err)

    def discover_pods(self):
        pod_name = ''
        if self.module.params['kube_file']:
            if HAS_YAML:
                with open(self.module.params['kube_file']) as f:
                    pod = yaml.safe_load(f)
                if 'metadata' in pod:
                    pod_name = pod['metadata'].get('name')
                else:
                    self.module.fail_json('No metadata in Kube file!\n%s' % pod)
            else:
                with open(self.module.params['kube_file']) as text:
                    re_pod_name = re.compile('^\\s{2,4}name: ["|\\\']?(?P<pod_name>[\\w|\\-|\\_]+)["|\\\']?', re.MULTILINE)
                    re_pod = re_pod_name.search(text.read())
                    if re_pod:
                        pod_name = re_pod.group(1)
        if not pod_name:
            self.module.fail_json("Deployment doesn't have a name!")
        all_pods = ''
        for name in ('name=%s$', 'name=%s-pod-*'):
            cmd = [self.executable, 'pod', 'ps', '-q', '--filter', name % pod_name]
            rc, out, err = self._command_run(cmd)
            all_pods += out
        ids = list(set([i for i in all_pods.splitlines() if i]))
        return ids

    def remove_associated_pods(self, pods):
        changed = False
        out_all, err_all = ('', '')
        for pod_id in pods:
            rc, out, err = self._command_run([self.executable, 'pod', 'rm', '-f', pod_id])
            if rc != 0:
                self.module.fail_json('Can NOT delete Pod %s' % pod_id)
            else:
                changed = True
                out_all += out
                err_all += err
        return (changed, out_all, err_all)

    def pod_recreate(self):
        pods = self.discover_pods()
        self.remove_associated_pods(pods)
        rc, out, err = self._command_run(self.command)
        if rc != 0:
            self.module.fail_json('Can NOT create Pod! Error: %s' % err)
        return (out, err)

    def play(self):
        rc, out, err = self._command_run(self.command)
        if rc != 0 and 'pod already exists' in err:
            if self.module.params['recreate']:
                out, err = self.pod_recreate()
                changed = True
            else:
                changed = False
            err = '\n'.join([i for i in err.splitlines() if 'pod already exists' not in i])
        elif rc != 0:
            self.module.fail_json(msg='Output: %s\nError=%s' % (out, err))
        else:
            changed = True
        return (changed, out, err)