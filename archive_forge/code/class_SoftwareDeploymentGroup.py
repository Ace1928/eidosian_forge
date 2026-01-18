import copy
import itertools
import uuid
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import output
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.heat import resource_group
from heat.engine.resources import signal_responder
from heat.engine import rsrc_defn
from heat.engine import software_config_io as swc_io
from heat.engine import support
from heat.rpc import api as rpc_api
class SoftwareDeploymentGroup(resource_group.ResourceGroup):
    """This resource associates a group of servers with some configuration.

    The configuration is to be deployed to all servers in the group.

    The properties work in a similar way to OS::Heat::SoftwareDeployment,
    and in addition to the attributes documented, you may pass any
    attribute supported by OS::Heat::SoftwareDeployment, including those
    exposing arbitrary outputs, and return a map of deployment names to
    the specified attribute.
    """
    support_status = support.SupportStatus(version='5.0.0')
    PROPERTIES = SERVERS, CONFIG, INPUT_VALUES, DEPLOY_ACTIONS, NAME, SIGNAL_TRANSPORT = ('servers', SoftwareDeployment.CONFIG, SoftwareDeployment.INPUT_VALUES, SoftwareDeployment.DEPLOY_ACTIONS, SoftwareDeployment.NAME, SoftwareDeployment.SIGNAL_TRANSPORT)
    ATTRIBUTES = STDOUTS, STDERRS, STATUS_CODES = ('deploy_stdouts', 'deploy_stderrs', 'deploy_status_codes')
    _ROLLING_UPDATES_SCHEMA_KEYS = MAX_BATCH_SIZE, PAUSE_TIME = (resource_group.ResourceGroup.MAX_BATCH_SIZE, resource_group.ResourceGroup.PAUSE_TIME)
    _sd_ps = SoftwareDeployment.properties_schema
    _rg_ps = resource_group.ResourceGroup.properties_schema
    properties_schema = {SERVERS: properties.Schema(properties.Schema.MAP, _('A map of names and server IDs to apply configuration to. The name is arbitrary and is used as the Heat resource name for the corresponding deployment.'), update_allowed=True, required=True), CONFIG: _sd_ps[CONFIG], INPUT_VALUES: _sd_ps[INPUT_VALUES], DEPLOY_ACTIONS: _sd_ps[DEPLOY_ACTIONS], NAME: _sd_ps[NAME], SIGNAL_TRANSPORT: _sd_ps[SIGNAL_TRANSPORT]}
    attributes_schema = {STDOUTS: attributes.Schema(_('A map of Nova names and captured stdouts from the configuration execution to each server.'), type=attributes.Schema.MAP), STDERRS: attributes.Schema(_('A map of Nova names and captured stderrs from the configuration execution to each server.'), type=attributes.Schema.MAP), STATUS_CODES: attributes.Schema(_('A map of Nova names and returned status code from the configuration execution.'), type=attributes.Schema.MAP)}
    rolling_update_schema = {MAX_BATCH_SIZE: properties.Schema(properties.Schema.INTEGER, _('The maximum number of deployments to replace at once.'), constraints=[constraints.Range(min=1)], default=1), PAUSE_TIME: properties.Schema(properties.Schema.NUMBER, _('The number of seconds to wait between batches of updates.'), constraints=[constraints.Range(min=0)], default=0)}
    update_policy_schema = {resource_group.ResourceGroup.ROLLING_UPDATE: properties.Schema(properties.Schema.MAP, schema=rolling_update_schema, support_status=support.SupportStatus(version='7.0.0')), resource_group.ResourceGroup.BATCH_CREATE: properties.Schema(properties.Schema.MAP, schema=resource_group.ResourceGroup.batch_create_schema, support_status=support.SupportStatus(version='7.0.0'))}

    def get_size(self):
        return len(self.properties[self.SERVERS])

    def _resource_names(self, size=None, update_rsrc_data=True):
        candidates = self.properties[self.SERVERS]
        if size is None:
            return iter(candidates)
        return itertools.islice(candidates, size)

    def res_def_changed(self, prop_diff):
        return True

    def _update_name_skiplist(self, properties):
        pass

    def _name_skiplist(self):
        return set()

    def get_resource_def(self, include_all=False):
        return dict(self.properties)

    def build_resource_definition(self, res_name, res_defn):
        props = copy.deepcopy(res_defn)
        servers = props.pop(self.SERVERS)
        props[SoftwareDeployment.SERVER] = servers.get(res_name)
        return rsrc_defn.ResourceDefinition(res_name, 'OS::Heat::SoftwareDeployment', props, None)

    def _member_attribute_name(self, key):
        if key == self.STDOUTS:
            n_attr = SoftwareDeployment.STDOUT
        elif key == self.STDERRS:
            n_attr = SoftwareDeployment.STDERR
        elif key == self.STATUS_CODES:
            n_attr = SoftwareDeployment.STATUS_CODE
        else:
            n_attr = key
        return n_attr

    def get_attribute(self, key, *path):
        rg = super(SoftwareDeploymentGroup, self)
        n_attr = self._member_attribute_name(key)
        rg_attr = rg.get_attribute(rg.ATTR_ATTRIBUTES, n_attr)
        return attributes.select_from_attribute(rg_attr, path)

    def _nested_output_defns(self, resource_names, get_attr_fn, get_res_fn):
        for attr in self.referenced_attrs():
            key = attr if isinstance(attr, str) else attr[0]
            n_attr = self._member_attribute_name(key)
            output_name = self._attribute_output_name(self.ATTR_ATTRIBUTES, n_attr)
            value = {r: get_attr_fn([r, n_attr]) for r in resource_names}
            yield output.OutputDefinition(output_name, value)

    def _try_rolling_update(self):
        if self.update_policy[self.ROLLING_UPDATE]:
            policy = self.update_policy[self.ROLLING_UPDATE]
            return self._replace(0, policy[self.MAX_BATCH_SIZE], policy[self.PAUSE_TIME])