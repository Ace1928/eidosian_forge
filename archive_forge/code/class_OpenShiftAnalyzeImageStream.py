from __future__ import (absolute_import, division, print_function)
from datetime import datetime
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
from ansible.module_utils.six import iteritems
class OpenShiftAnalyzeImageStream(object):

    def __init__(self, ignore_invalid_refs, max_creation_timestamp, module):
        self.max_creationTimestamp = max_creation_timestamp
        self.used_tags = {}
        self.used_images = {}
        self.ignore_invalid_refs = ignore_invalid_refs
        self.module = module

    def analyze_reference_image(self, image, referrer):
        result, error = parse_docker_image_ref(image, self.module)
        if error:
            return error
        if not result['hostname'] or not result['namespace']:
            return None
        if not result['digest']:
            if result['tag'] == '':
                result['tag'] = 'latest'
            key = '%s/%s:%s' % (result['namespace'], result['name'], result['tag'])
            if key not in self.used_tags:
                self.used_tags[key] = []
            self.used_tags[key].append(referrer)
        else:
            key = '%s/%s@%s' % (result['namespace'], result['name'], result['digest'])
            if key not in self.used_images:
                self.used_images[key] = []
            self.used_images[key].append(referrer)

    def analyze_refs_from_pod_spec(self, podSpec, referrer):
        for container in podSpec.get('initContainers', []) + podSpec.get('containers', []):
            image = container.get('image')
            if len(image.strip()) == 0:
                continue
            err = self.analyze_reference_image(image, referrer)
            if err:
                return err
        return None

    def analyze_refs_from_pods(self, pods):
        for pod in pods:
            too_young = is_too_young_object(pod, self.max_creationTimestamp)
            if pod['status']['phase'] not in ('Running', 'Pending') and too_young:
                continue
            referrer = {'kind': pod['kind'], 'namespace': pod['metadata']['namespace'], 'name': pod['metadata']['name']}
            err = self.analyze_refs_from_pod_spec(pod['spec'], referrer)
            if err:
                return err
        return None

    def analyze_refs_pod_creators(self, resources):
        keys = ('ReplicationController', 'DeploymentConfig', 'DaemonSet', 'Deployment', 'ReplicaSet', 'StatefulSet', 'Job', 'CronJob')
        for k, objects in iteritems(resources):
            if k not in keys:
                continue
            for obj in objects:
                if k == 'CronJob':
                    spec = obj['spec']['jobTemplate']['spec']['template']['spec']
                else:
                    spec = obj['spec']['template']['spec']
                referrer = {'kind': obj['kind'], 'namespace': obj['metadata']['namespace'], 'name': obj['metadata']['name']}
                err = self.analyze_refs_from_pod_spec(spec, referrer)
                if err:
                    return err
        return None

    def analyze_refs_from_strategy(self, build_strategy, namespace, referrer):

        def _determine_source_strategy():
            for src in ('sourceStrategy', 'dockerStrategy', 'customStrategy'):
                strategy = build_strategy.get(src)
                if strategy:
                    return strategy.get('from')
            return None

        def _parse_image_stream_image_name(name):
            v = name.split('@')
            if len(v) != 2:
                return (None, None, 'expected exactly one @ in the isimage name %s' % name)
            name = v[0]
            tag = v[1]
            if len(name) == 0 or len(tag) == 0:
                return (None, None, 'image stream image name %s must have a name and ID' % name)
            return (name, tag, None)

        def _parse_image_stream_tag_name(name):
            if '@' in name:
                return (None, None, '%s is an image stream image, not an image stream tag' % name)
            v = name.split(':')
            if len(v) != 2:
                return (None, None, 'expected exactly one : delimiter in the istag %s' % name)
            name = v[0]
            tag = v[1]
            if len(name) == 0 or len(tag) == 0:
                return (None, None, 'image stream tag name %s must have a name and a tag' % name)
            return (name, tag, None)
        from_strategy = _determine_source_strategy()
        if from_strategy:
            if from_strategy.get('kind') == 'DockerImage':
                docker_image_ref = from_strategy.get('name').strip()
                if len(docker_image_ref) > 0:
                    err = self.analyze_reference_image(docker_image_ref, referrer)
            elif from_strategy.get('kind') == 'ImageStreamImage':
                name, tag, error = _parse_image_stream_image_name(from_strategy.get('name'))
                if error:
                    if not self.ignore_invalid_refs:
                        return error
                else:
                    namespace = from_strategy.get('namespace') or namespace
                    self.used_images.append({'namespace': namespace, 'name': name, 'tag': tag})
            elif from_strategy.get('kind') == 'ImageStreamTag':
                name, tag, error = _parse_image_stream_tag_name(from_strategy.get('name'))
                if error:
                    if not self.ignore_invalid_refs:
                        return error
                else:
                    namespace = from_strategy.get('namespace') or namespace
                    self.used_tags.append({'namespace': namespace, 'name': name, 'tag': tag})

    def analyze_refs_from_build_strategy(self, resources):
        keys = ('BuildConfig', 'Build')
        for k, objects in iteritems(resources):
            if k not in keys:
                continue
            for obj in objects:
                referrer = {'kind': obj['kind'], 'namespace': obj['metadata']['namespace'], 'name': obj['metadata']['name']}
                error = self.analyze_refs_from_strategy(obj['spec']['strategy'], obj['metadata']['namespace'], referrer)
                if error is not None:
                    return '%s/%s/%s: %s' % (referrer['kind'], referrer['namespace'], referrer['name'], error)

    def analyze_image_stream(self, resources):
        error = self.analyze_refs_from_pods(resources['Pod'])
        if error:
            return (None, None, error)
        error = self.analyze_refs_pod_creators(resources)
        if error:
            return (None, None, error)
        error = self.analyze_refs_from_build_strategy(resources)
        return (self.used_tags, self.used_images, error)