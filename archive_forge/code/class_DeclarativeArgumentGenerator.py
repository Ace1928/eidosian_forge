from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_property
class DeclarativeArgumentGenerator(object):
    """An argument generator that operates off a declarative configuration.

  When using this generator, you must provide attributes for the arguments that
  should be generated. All resource arguments must be provided and arguments
  will only be generated for API fields for which attributes were provided.
  """

    def __init__(self, arg_info):
        """Creates a new Argument Generator.

    Args:
      arg_info: [yaml_arg_schema.Argument], Information about
        request fields and how to map them into arguments.
    """
        self.arg_info = arg_info
        self.resource_args = _GetResources(self.arg_info)

    def GenerateArgs(self, methods):
        """Generates all the CLI arguments required to call this method.

    Args:
      methods: list[APIMethod], list of methods to generate arguments for.

    Returns:
      {str, calliope.base.Action}, A map of field name to the argument.
    """
        shared_attribute_resource_dict = _GetSharedAttributes(self.resource_args)
        shared_resource_attributes_list = list(shared_attribute_resource_dict)
        args = [arg.Generate(methods, shared_resource_attributes_list) for arg in self.arg_info]
        primary_resource_args = _GetMethodResourceArgs(self.resource_args, methods)
        primary_names = set((arg.primary_resource and arg.primary_resource.name for arg in primary_resource_args))
        for attribute, resource_args in shared_attribute_resource_dict.items():
            resource_names = list(set(resource_args))
            resource_names.sort(key=lambda name: '' if name in primary_names else name)
            args.append(base.Argument('--' + attribute, help="For resources [{}], provides fallback value for resource {attr} attribute. When the resource's full URI path is not provided, {attr} will fallback to this flag value.".format(', '.join(resource_names), attr=attribute)))
        return args

    def GetPrimaryResource(self, methods, namespace):
        """Gets primary resource based on user input and returns single method.

    This determines which api method to use to make api request. If there
    is only one potential request method, return the one request method.

    Args:
      methods: list[APIMethod], The method to generate arguments for.
      namespace: The argparse namespace.

    Returns:
      MethodResourceArg, gets the primary resource arg and method the
        user specified in the namespace.

    Raises:
      ConflictingResourcesError: occurs when user specifies too many primary
        resources.
    """
        specified_methods = []
        primary_resources = _GetMethodResourceArgs(self.resource_args, methods)
        if not primary_resources:
            return MethodResourceArg(primary_resource=None, method=None)
        elif len(primary_resources) == 1:
            return primary_resources.pop()
        for method_info in primary_resources:
            method = method_info.method
            primary_resource = method_info.primary_resource
            if not method or not primary_resource:
                raise util.InvalidSchemaError('If more than one request collection is specified, a resource argument that corresponds with the collection, must be specified in YAML command.')
            method_collection = _GetCollectionName(method, is_parent=primary_resource.is_parent_resource)
            specified_resource = method_info.Parse(namespace)
            primary_collection = specified_resource and specified_resource.GetCollectionInfo().full_name
            if method_collection == primary_collection:
                specified_methods.append(method_info)
        if len(specified_methods) > 1:
            uris = []
            for method_info in specified_methods:
                if (parsed := method_info.Parse(namespace)):
                    uris.append(parsed.RelativeName())
            args = ', '.join(uris)
            raise ConflictingResourcesError(f'User specified multiple primary resource arguments: [{args}]. Unable to determine api request method.')
        if len(specified_methods) == 1:
            return specified_methods.pop()
        else:
            return MethodResourceArg(primary_resource=None, method=None)

    def CreateRequest(self, namespace, method, static_fields=None, labels=None, command_type=None, existing_message=None):
        """Generates the request object for the method call from the parsed args.

    Args:
      namespace: The argparse namespace.
      method: APIMethod, api method used to make request message.
      static_fields: {str, value}, A mapping of API field name to value to
        insert into the message. This is a convenient way to insert extra data
        while the request is being constructed for fields that don't have
        corresponding arguments.
      labels: The labels section of the command spec.
      command_type: Type of the command, i.e. CREATE, UPDATE.
      existing_message: the apitools message returned from server, which is used
        to construct the to-be-modified message when the command follows
        get-modify-update pattern.

    Returns:
      The apitools message to be send to the method.
    """
        new_message = method.GetRequestType()()
        if existing_message:
            message = arg_utils.ParseExistingMessageIntoMessage(new_message, existing_message, method)
        else:
            message = new_message
        if labels:
            if command_type == yaml_command_schema.CommandType.CREATE:
                _ParseLabelsIntoCreateMessage(message, namespace, labels.api_field)
            elif command_type == yaml_command_schema.CommandType.UPDATE:
                need_update = _ParseLabelsIntoUpdateMessage(message, namespace, labels.api_field)
                if need_update:
                    update_mask_path = update.GetMaskFieldPath(method)
                    _AddLabelsToUpdateMask(static_fields, update_mask_path)
        arg_utils.ParseStaticFieldsIntoMessage(message, static_fields=static_fields)
        for arg in self.arg_info:
            arg.Parse(method, message, namespace)
        return message

    def GetResponseResourceRef(self, id_value, namespace, method):
        """Gets a resource reference for a resource returned by a list call.

    It parses the namespace to find a reference to the parent collection and
    then creates a reference to the child resource with the given id_value.

    Args:
      id_value: str, The id of the child resource that was returned.
      namespace: The argparse namespace.
      method: APIMethod, method used to make the api request

    Returns:
      resources.Resource, The parsed resource reference.
    """
        methods = [method] if method else []
        parent_ref = self.GetPrimaryResource(methods, namespace).Parse(namespace)
        return resources.REGISTRY.Parse(id_value, collection=method.collection.full_name, api_version=method.collection.api_version, params=parent_ref.AsDict())

    def Limit(self, namespace):
        """Gets the value of the limit flag (if present)."""
        return getattr(namespace, 'limit', None)

    def PageSize(self, namespace):
        """Gets the value of the page size flag (if present)."""
        return getattr(namespace, 'page_size', None)