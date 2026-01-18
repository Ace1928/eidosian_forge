import copy
import functools
import json
import os
import urllib.request
import glance_store as store_api
from glance_store import backend
from glance_store import exceptions as store_exceptions
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import encodeutils
from oslo_utils import excutils
from oslo_utils import timeutils
from oslo_utils import units
import taskflow
from taskflow.patterns import linear_flow as lf
from taskflow import retry
from taskflow import task
from glance.api import common as api_common
import glance.async_.flows._internal_plugins as internal_plugins
import glance.async_.flows.plugins as import_plugins
from glance.async_ import utils
from glance.common import exception
from glance.common.scripts.image_import import main as image_import
from glance.common.scripts import utils as script_utils
from glance.common import store_utils
from glance.i18n import _, _LE, _LI
from glance.quota import keystone as ks_quota
class _ImportMetadata(task.Task):
    default_provides = 'image_size'

    def __init__(self, task_id, task_type, context, action_wrapper, import_req):
        self.task_id = task_id
        self.task_type = task_type
        self.context = context
        self.action_wrapper = action_wrapper
        self.import_req = import_req
        self.props_to_copy = CONF.glance_download_properties.extra_properties
        self.properties = {}
        self.old_properties = {}
        self.old_attributes = {}
        super(_ImportMetadata, self).__init__(name='%s-ImportMetdata-%s' % (task_type, task_id))

    def execute(self):
        try:
            glance_endpoint = utils.get_glance_endpoint(self.context, self.import_req['method']['glance_region'], self.import_req['method']['glance_service_interface'])
            glance_image_id = self.import_req['method']['glance_image_id']
            image_download_metadata_url = '%s/v2/images/%s' % (glance_endpoint, glance_image_id)
            LOG.info(_LI('Fetching glance image metadata from remote host %s'), image_download_metadata_url)
            token = self.context.auth_token
            request = urllib.request.Request(image_download_metadata_url, headers={'X-Auth-Token': token})
            with urllib.request.urlopen(request) as payload:
                data = json.loads(payload.read().decode('utf-8'))
            if data.get('status') != 'active':
                raise _InvalidGlanceDownloadImageStatus(_('Source image status should be active instead of %s') % data['status'])
            for key, value in data.items():
                for metadata in self.props_to_copy:
                    if key.startswith(metadata):
                        self.properties[key] = value
            with self.action_wrapper as action:
                self.old_properties = action.image_extra_properties
                self.old_attributes = {'container_format': action.image_container_format, 'disk_format': action.image_disk_format}
                action.set_image_attribute(disk_format=data['disk_format'], container_format=data['container_format'])
                if self.properties:
                    action.set_image_extra_properties(self.properties)
            try:
                return int(data['size'])
            except (ValueError, KeyError):
                raise exception.ImportTaskError(_('Size attribute of remote image %s could not be determined.' % glance_image_id))
        except Exception as e:
            with excutils.save_and_reraise_exception():
                LOG.error('Task %(task_id)s failed with exception %(error)s', {'error': encodeutils.exception_to_unicode(e), 'task_id': self.task_id})

    def revert(self, result, **kwargs):
        """Revert the extra properties set and set the image in queued"""
        with self.action_wrapper as action:
            for image_property in self.properties:
                if image_property not in self.old_properties:
                    action.pop_extra_property(image_property)
            action.set_image_extra_properties(self.old_properties)
            action.set_image_attribute(status='queued', **self.old_attributes)