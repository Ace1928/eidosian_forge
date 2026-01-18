from __future__ import absolute_import, division, print_function
import copy
import time
import traceback
from datetime import datetime
from ansible_collections.kubernetes.core.plugins.module_utils.ansiblemodule import (
from ansible_collections.kubernetes.core.plugins.module_utils.args_common import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.client import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.core import (
from ansible_collections.kubernetes.core.plugins.module_utils.k8s.exceptions import (
from ansible.module_utils._text import to_native
class K8sDrainAnsible(object):

    def __init__(self, module, client):
        self._module = module
        self._api_instance = core_v1_api.CoreV1Api(client.client)
        self._drain_options = module.params.get('delete_options', {})
        self._delete_options = None
        if self._drain_options.get('terminate_grace_period'):
            self._delete_options = V1DeleteOptions(grace_period_seconds=self._drain_options.get('terminate_grace_period'))
        self._changed = False

    def wait_for_pod_deletion(self, pods, wait_timeout, wait_sleep):
        start = datetime.now()

        def _elapsed_time():
            return (datetime.now() - start).seconds
        response = None
        pod = pods.pop()
        while (_elapsed_time() < wait_timeout or wait_timeout == 0) and pods:
            if not pod:
                pod = pods.pop()
            try:
                response = self._api_instance.read_namespaced_pod(namespace=pod[0], name=pod[1])
                if not response:
                    pod = None
                time.sleep(wait_sleep)
            except ApiException as exc:
                if exc.reason != 'Not Found':
                    self._module.fail_json(msg='Exception raised: {0}'.format(exc.reason))
                pod = None
            except Exception as e:
                self._module.fail_json(msg='Exception raised: {0}'.format(to_native(e)))
        if not pods:
            return None
        return 'timeout reached while pods were still running.'

    def evict_pods(self, pods):
        for namespace, name in pods:
            try:
                if self._drain_options.get('disable_eviction'):
                    self._api_instance.delete_namespaced_pod(name=name, namespace=namespace, body=self._delete_options)
                else:
                    body = v1_eviction(delete_options=self._delete_options, metadata=V1ObjectMeta(name=name, namespace=namespace))
                    self._api_instance.create_namespaced_pod_eviction(name=name, namespace=namespace, body=body)
                self._changed = True
            except ApiException as exc:
                if exc.reason != 'Not Found':
                    self._module.fail_json(msg='Failed to delete pod {0}/{1} due to: {2}'.format(namespace, name, exc.reason))
            except Exception as exc:
                self._module.fail_json(msg='Failed to delete pod {0}/{1} due to: {2}'.format(namespace, name, to_native(exc)))

    def delete_or_evict_pods(self, node_unschedulable):
        result = []
        if not node_unschedulable:
            self.patch_node(unschedulable=True)
            result.append('node {0} marked unschedulable.'.format(self._module.params.get('name')))
            self._changed = True
        else:
            result.append('node {0} already marked unschedulable.'.format(self._module.params.get('name')))

        def _revert_node_patch():
            if self._changed:
                self._changed = False
                self.patch_node(unschedulable=False)
        try:
            field_selector = 'spec.nodeName={name}'.format(name=self._module.params.get('name'))
            pod_list = self._api_instance.list_pod_for_all_namespaces(field_selector=field_selector)
            force = self._drain_options.get('force', False)
            ignore_daemonset = self._drain_options.get('ignore_daemonsets', False)
            delete_emptydir_data = self._drain_options.get('delete_emptydir_data', False)
            pods, warnings, errors = filter_pods(pod_list.items, force, ignore_daemonset, delete_emptydir_data)
            if errors:
                _revert_node_patch()
                self._module.fail_json(msg='Pod deletion errors: {0}'.format(' '.join(errors)))
        except ApiException as exc:
            if exc.reason != 'Not Found':
                _revert_node_patch()
                self._module.fail_json(msg='Failed to list pod from node {name} due to: {reason}'.format(name=self._module.params.get('name'), reason=exc.reason), status=exc.status)
            pods = []
        except Exception as exc:
            _revert_node_patch()
            self._module.fail_json(msg='Failed to list pod from node {name} due to: {error}'.format(name=self._module.params.get('name'), error=to_native(exc)))
        if pods:
            self.evict_pods(pods)
            number_pod = len(pods)
            if self._drain_options.get('wait_timeout') is not None:
                warn = self.wait_for_pod_deletion(pods, self._drain_options.get('wait_timeout'), self._drain_options.get('wait_sleep'))
                if warn:
                    warnings.append(warn)
            result.append('{0} Pod(s) deleted from node.'.format(number_pod))
        if warnings:
            return dict(result=' '.join(result), warnings=warnings)
        return dict(result=' '.join(result))

    def patch_node(self, unschedulable):
        body = {'spec': {'unschedulable': unschedulable}}
        try:
            self._api_instance.patch_node(name=self._module.params.get('name'), body=body)
        except Exception as exc:
            self._module.fail_json(msg='Failed to patch node due to: {0}'.format(to_native(exc)))

    def execute_module(self):
        state = self._module.params.get('state')
        name = self._module.params.get('name')
        try:
            node = self._api_instance.read_node(name=name)
        except ApiException as exc:
            if exc.reason == 'Not Found':
                self._module.fail_json(msg='Node {0} not found.'.format(name))
            self._module.fail_json(msg="Failed to retrieve node '{0}' due to: {1}".format(name, exc.reason), status=exc.status)
        except Exception as exc:
            self._module.fail_json(msg="Failed to retrieve node '{0}' due to: {1}".format(name, to_native(exc)))
        result = {}
        if state == 'cordon':
            if node.spec.unschedulable:
                self._module.exit_json(result='node {0} already marked unschedulable.'.format(name))
            self.patch_node(unschedulable=True)
            result['result'] = 'node {0} marked unschedulable.'.format(name)
            self._changed = True
        elif state == 'uncordon':
            if not node.spec.unschedulable:
                self._module.exit_json(result='node {0} already marked schedulable.'.format(name))
            self.patch_node(unschedulable=False)
            result['result'] = 'node {0} marked schedulable.'.format(name)
            self._changed = True
        else:
            ret = self.delete_or_evict_pods(node_unschedulable=node.spec.unschedulable)
            result.update(ret)
        self._module.exit_json(changed=self._changed, **result)