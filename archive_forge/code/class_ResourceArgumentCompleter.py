from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.util import resource as resource_lib  # pylint: disable=unused-import
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.concepts import resource_parameter_info
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
class ResourceArgumentCompleter(completers.ResourceCompleter):
    """A completer for an argument that's part of a resource argument."""

    def __init__(self, resource_spec, collection_info, method, static_params=None, id_field=None, param=None, **kwargs):
        """Initializes."""
        self.resource_spec = resource_spec
        self._method = method
        self._static_params = static_params or {}
        self.id_field = id_field or DEFAULT_ID_FIELD
        collection_name = collection_info.full_name
        api_version = collection_info.api_version
        super(ResourceArgumentCompleter, self).__init__(collection=collection_name, api_version=api_version, param=param, parse_all=True, **kwargs)

    @property
    def method(self):
        """Gets the list method for the collection.

    Returns:
      googlecloudsdk.command_lib.util.apis.registry.APIMethod, the method.
    """
        return self._method

    def _ParentParams(self):
        """Get the parent params of the collection."""
        return self.collection_info.GetParams('')[:-1]

    def _GetUpdaters(self):
        """Helper function to build dict of updaters."""
        final_param = self.collection_info.GetParams('')[-1]
        for i, attribute in enumerate(self.resource_spec.attributes):
            if self.resource_spec.ParamName(attribute.name) == final_param:
                attribute_idx = i
                break
        else:
            attribute_idx = 0
        updaters = {}
        for i, attribute in enumerate(self.resource_spec.attributes[:attribute_idx]):
            completer = CompleterForAttribute(self.resource_spec, attribute.name)
            if completer:
                updaters[self.resource_spec.ParamName(attribute.name)] = (completer, True)
            else:
                updaters[self.resource_spec.ParamName(attribute.name)] = (None, False)
        return updaters

    def ParameterInfo(self, parsed_args, argument):
        """Builds a ResourceParameterInfo object.

    Args:
      parsed_args: the namespace.
      argument: unused.

    Returns:
      ResourceParameterInfo, the parameter info for runtime information.
    """
        resource_info = parsed_args.CONCEPTS.ArgNameToConceptInfo(argument.dest)
        updaters = self._GetUpdaters()
        return resource_parameter_info.ResourceParameterInfo(resource_info, parsed_args, argument, updaters=updaters, collection=self.collection)

    def ValidateAttributeSources(self, aggregations):
        """Validates that parent attributes values exitst before making request."""
        parameters_needing_resolution = set([p.name for p in self.parameters[:-1]])
        resolved_parameters = set([a.name for a in aggregations])
        for attribute in self.resource_spec.attributes:
            if CompleterForAttribute(self.resource_spec, attribute.name):
                resolved_parameters.add(self.resource_spec.attribute_to_params_map[attribute.name])
        return parameters_needing_resolution.issubset(resolved_parameters)

    def Update(self, parameter_info, aggregations):
        if self.method is None:
            return None
        if not self.ValidateAttributeSources(aggregations):
            return None
        log.info('Cache query parameters={} aggregations={}resource info={}'.format([(p, parameter_info.GetValue(p)) for p in self.collection_info.GetParams('')], [(p.name, p.value) for p in aggregations], parameter_info.resource_info.attribute_to_args_map))
        parent_translator = self._GetParentTranslator(parameter_info, aggregations)
        try:
            query = self.BuildListQuery(parameter_info, aggregations, parent_translator=parent_translator)
        except Exception as e:
            if properties.VALUES.core.print_completion_tracebacks.GetBool():
                raise
            log.info(six.text_type(e).rstrip())
            raise Error('Could not build query to list completions: {} {}'.format(type(e), six.text_type(e).rstrip()))
        try:
            response = self.method.Call(query)
            response_collection = self.method.collection
            items = [self._ParseResponse(r, response_collection, parameter_info=parameter_info, aggregations=aggregations, parent_translator=parent_translator) for r in response]
            log.info('cache items={}'.format([i.RelativeName() for i in items]))
        except Exception as e:
            if properties.VALUES.core.print_completion_tracebacks.GetBool():
                raise
            log.info(six.text_type(e).rstrip())
            if isinstance(e, messages.ValidationError):
                raise Error('Update query failed, may not have enough information to list existing resources: {} {}'.format(type(e), six.text_type(e).rstrip()))
            raise Error('Update query [{}]: {} {}'.format(query, type(e), six.text_type(e).rstrip()))
        return [self.StringToRow(item.RelativeName()) for item in items]

    def _ParseResponse(self, response, response_collection, parameter_info=None, aggregations=None, parent_translator=None):
        """Gets a resource ref from a single item in a list response."""
        param_values = self._GetParamValuesFromParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
        param_names = response_collection.detailed_params
        for param in param_names:
            val = getattr(response, param, None)
            if val is not None:
                param_values[param] = val
        line = getattr(response, self.id_field, '')
        return resources.REGISTRY.Parse(line, collection=response_collection.full_name, params=param_values)

    def _GetParamValuesFromParent(self, parameter_info, aggregations=None, parent_translator=None):
        parent_ref = self.GetParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
        if not parent_ref:
            return {}
        params = parent_ref.AsDict()
        if parent_translator:
            return parent_translator.ToChildParams(params)
        return params

    def _GetAggregationsValuesDict(self, aggregations):
        """Build a {str: str} dict of name to value for aggregations."""
        aggregations_dict = {}
        aggregations = [] if aggregations is None else aggregations
        for aggregation in aggregations:
            if aggregation.value:
                aggregations_dict[aggregation.name] = aggregation.value
        return aggregations_dict

    def BuildListQuery(self, parameter_info, aggregations=None, parent_translator=None):
        """Builds a list request to list values for the given argument.

    Args:
      parameter_info: the runtime ResourceParameterInfo object.
      aggregations: a list of _RuntimeParameter objects.
      parent_translator: a ParentTranslator object if needed.

    Returns:
      The apitools request.
    """
        method = self.method
        if method is None:
            return None
        message = method.GetRequestType()()
        for field, value in six.iteritems(self._static_params):
            arg_utils.SetFieldInMessage(message, field, value)
        parent = self.GetParent(parameter_info, aggregations=aggregations, parent_translator=parent_translator)
        if not parent:
            return message
        message_resource_map = {}
        if parent_translator:
            message_resource_map = parent_translator.MessageResourceMap(message, parent)
        arg_utils.ParseResourceIntoMessage(parent, method, message, message_resource_map=message_resource_map, is_primary_resource=True)
        return message

    def _GetParentTranslator(self, parameter_info, aggregations=None):
        """Get a special parent translator if needed and available."""
        aggregations_dict = self._GetAggregationsValuesDict(aggregations)
        param_values = self._GetRawParamValuesForParent(parameter_info, aggregations_dict=aggregations_dict)
        try:
            self._ParseDefaultParent(param_values)
            return None
        except resources.ParentCollectionResolutionException:
            key = '.'.join(self._ParentParams())
            if key in _PARENT_TRANSLATORS:
                return _PARENT_TRANSLATORS.get(key)
        except resources.Error:
            return None

    def _GetRawParamValuesForParent(self, parameter_info, aggregations_dict=None):
        """Get raw param values for the resource in prep for parsing parent."""
        param_values = {p: parameter_info.GetValue(p) for p in self._ParentParams()}
        for name, value in six.iteritems(aggregations_dict or {}):
            if value and (not param_values.get(name, None)):
                param_values[name] = value
        final_param = self.collection_info.GetParams('')[-1]
        if param_values.get(final_param, None) is None:
            param_values[final_param] = 'fake'
        return param_values

    def _ParseDefaultParent(self, param_values):
        """Parse the parent for a resource using default collection."""
        resource = resources.Resource(resources.REGISTRY, collection_info=self.collection_info, subcollection='', param_values=param_values, endpoint_url=None)
        return resource.Parent()

    def GetParent(self, parameter_info, aggregations=None, parent_translator=None):
        """Gets the parent reference of the parsed parameters.

    Args:
      parameter_info: the runtime ResourceParameterInfo object.
      aggregations: a list of _RuntimeParameter objects.
      parent_translator: a ParentTranslator for translating to a special
        parent collection, if needed.

    Returns:
      googlecloudsdk.core.resources.Resource | None, the parent resource or None
        if no parent was found.
    """
        aggregations_dict = self._GetAggregationsValuesDict(aggregations)
        param_values = self._GetRawParamValuesForParent(parameter_info, aggregations_dict=aggregations_dict)
        try:
            if not parent_translator:
                return self._ParseDefaultParent(param_values)
            return parent_translator.Parse(self._ParentParams(), parameter_info, aggregations_dict)
        except resources.ParentCollectionResolutionException as e:
            log.info(six.text_type(e).rstrip())
            return None
        except resources.Error as e:
            log.info(six.text_type(e).rstrip())
            return None

    def __eq__(self, other):
        """Overrides."""
        if not isinstance(other, ResourceArgumentCompleter):
            return False
        return self.resource_spec == other.resource_spec and self.collection == other.collection and (self.method == other.method)