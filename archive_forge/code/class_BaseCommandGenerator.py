from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class BaseCommandGenerator(six.with_metaclass(abc.ABCMeta, object)):
    """Base class for command generation."""

    def __init__(self, spec):
        self.spec = spec
        self.has_request_method = yaml_command_schema.CommandType.HasRequestMethod(spec.command_type)
        self.InitializeGeneratorForCommand()

    def InitializeGeneratorForCommand(self):
        """Initializes the arg_generator for command."""
        from googlecloudsdk.command_lib.util.apis import arg_marshalling
        if self.has_request_method:
            self.methods = self._GetMethods()
        else:
            self.methods = []
        self.arg_generator = arg_marshalling.DeclarativeArgumentGenerator(self.spec.arguments.params)

    def _GetMethods(self, method=None):
        methods = []
        for collection in self.spec.request.collections:
            methods.append(registry.GetMethod(collection, method or self.spec.request.method, self.spec.request.api_version, disable_pagination=self.spec.request.disable_pagination))
        return methods

    def _CommonArgs(self, parser):
        """Performs argument actions common to all commands.

    Adds all generated arguments to the parser
    Sets the command output format if specified

    Args:
      parser: The argparse parser.
    """
        args = self.arg_generator.GenerateArgs(self.methods)
        parser = self._Exclude(parser)
        for arg in args:
            arg.AddToParser(parser)
        if self.spec.arguments.additional_arguments_hook:
            for arg in self.spec.arguments.additional_arguments_hook():
                arg.AddToParser(parser)
        if self.spec.output.format:
            parser.display_info.AddFormat(self.spec.output.format)
        if self.spec.output.flatten:
            parser.display_info.AddFlatten(self.spec.output.flatten)

    def _RegisterURIFunc(self, args):
        """Generates and registers a function to create a URI from a resource.

    Args:
      args: The argparse namespace.

    Returns:
      f(resource) -> str, A function that converts the given resource payload
      into a URI.
    """

        def URIFunc(resource):
            id_value = getattr(resource, self.spec.response.id_field)
            method = self.arg_generator.GetPrimaryResource(self.methods, args).method
            ref = self.arg_generator.GetResponseResourceRef(id_value, args, method)
            return ref.SelfLink()
        if hasattr(args, 'uri'):
            args.GetDisplayInfo().AddUriFunc(URIFunc)

    def _Exclude(self, parser):
        """Excludes specified arguments from the parser.

    Args:
      parser: The argparse parser.

    Returns:
      The argparse parser.
    """
        for arg in self.spec.arguments.exclude:
            base.Argument('--{}'.format(arg), help='').RemoveFromParser(parser)
        return parser

    def _GetRuntimeMethods(self, args):
        if not self.spec.request.modify_method_hook:
            return self.methods
        specified = self.arg_generator.GetPrimaryResource(self.methods, args)
        ref = specified.Parse(args)
        new_method_name = self.spec.request.modify_method_hook(ref, args)
        return self._GetMethods(new_method_name)

    def _CommonRun(self, args, existing_message=None, update_mask=None):
        """Performs run actions common to all commands.

    Parses the resource argument into a resource reference
    Prompts the user to continue (if applicable)
    Calls the API method with the request generated from the parsed arguments

    Args:
      args: The argparse parser.
      existing_message: the apitools message returned from previous request.
      update_mask: auto-generated mask from updated fields.

    Returns:
      (resources.Resource, response), A tuple of the parsed resource reference
      and the API response from the method call.
    """
        self.methods = self._GetRuntimeMethods(args)
        resource = self.arg_generator.GetPrimaryResource(self.methods, args)
        method = resource.method
        ref = resource.Parse(args)
        if self.spec.input.confirmation_prompt:
            console_io.PromptContinue(message=self._Format(self.spec.input.confirmation_prompt, ref, self._GetDisplayResourceType(args), self._GetDisplayName(ref, args)), default=self.spec.input.default_continue, throw_if_unattended=True, cancel_on_no=True)
        if self.spec.request.issue_request_hook:
            return (ref, self.spec.request.issue_request_hook(ref, args))
        if self.spec.request.create_request_hook:
            request = self.spec.request.create_request_hook(ref, args)
        else:
            static_fields = {}
            if update_mask:
                static_fields.update(update_mask)
            if self.spec.request.static_fields:
                static_fields.update(self.spec.request.static_fields)
            request = self.arg_generator.CreateRequest(args, method, static_fields, self.spec.arguments.labels, self.spec.command_type, existing_message=existing_message)
            for hook in self.spec.request.modify_request_hooks:
                request = hook(ref, args, request)
        response = method.Call(request, limit=self.arg_generator.Limit(args), page_size=self.arg_generator.PageSize(args))
        return (ref, response)

    def _Format(self, format_string, resource_ref, display_type, display_name=None):
        return yaml_command_schema_util.FormatResourceAttrStr(format_string, resource_ref, display_name, display_type)

    def _GetDisplayName(self, resource_ref, args):
        primary_resource_arg = self.arg_generator.GetPrimaryResource(self.methods, args).primary_resource
        if primary_resource_arg and primary_resource_arg.display_name_hook:
            return primary_resource_arg.display_name_hook(resource_ref, args)
        return resource_ref.Name() if resource_ref else None

    def _GetDisplayResourceType(self, args):
        if (spec_display := self.spec.request.display_resource_type):
            return spec_display
        primary_resource_arg = self.arg_generator.GetPrimaryResource(self.methods, args).primary_resource
        if primary_resource_arg and (not primary_resource_arg.is_parent_resource):
            return primary_resource_arg.name
        else:
            return None

    def _HandleResponse(self, response, args=None):
        """Process the API response.

    Args:
      response: The apitools message object containing the API response.
      args: argparse.Namespace, The parsed args.

    Raises:
      core.exceptions.Error: If an error was detected and extracted from the
        response.

    Returns:
      A possibly modified response.
    """
        if self.spec.response.error:
            error = self._FindPopulatedAttribute(response, self.spec.response.error.field.split('.'))
            if error:
                messages = []
                if self.spec.response.error.code:
                    messages.append('Code: [{}]'.format(_GetAttribute(error, self.spec.response.error.code)))
                if self.spec.response.error.message:
                    messages.append('Message: [{}]'.format(_GetAttribute(error, self.spec.response.error.message)))
                if messages:
                    raise exceptions.Error(' '.join(messages))
                raise exceptions.Error(six.text_type(error))
        if self.spec.response.result_attribute:
            response = _GetAttribute(response, self.spec.response.result_attribute)
        for hook in self.spec.response.modify_response_hooks:
            response = hook(response, args)
        return response

    def _GetOperationRef(self, operation):
        for i, collection in enumerate(self.spec.async_.collections):
            try:
                resource = resources.REGISTRY.Parse(getattr(operation, self.spec.async_.response_name_field), collection=collection, api_version=self.spec.async_.api_version or self.spec.request.api_version)
                return (resource, collection)
            except resources.UserError as e:
                if i == len(self.spec.async_.collections) - 1:
                    raise e

    def _HandleAsync(self, args, resource_ref, operation, request_string, extract_resource_result=True):
        """Handles polling for operations if the async flag is provided.

    Args:
      args: argparse.Namespace, The parsed args.
      resource_ref: resources.Resource, The resource reference for the resource
        being operated on (not the operation itself)
      operation: The operation message response.
      request_string: The format string to print indicating a request has been
        issued for the resource. If None, nothing is printed.
      extract_resource_result: bool, True to return the original resource as
        the result or False to just return the operation response when it is
        done. You would set this to False for things like Delete where the
        resource no longer exists when the operation is done.

    Returns:
      The response (either the operation or the original resource).
    """
        operation_ref, operation_collection = self._GetOperationRef(operation)
        request_string = self.spec.async_.request_issued_message or request_string
        if request_string:
            log.status.Print(self._Format(request_string, resource_ref, self._GetDisplayResourceType(args), self._GetDisplayName(resource_ref, args)))
        if args.async_:
            log.status.Print(self._Format('Check operation [{{{}}}] for status.'.format(yaml_command_schema_util.REL_NAME_FORMAT_KEY), operation_ref, self._GetDisplayResourceType(args)))
            return operation
        method = self.arg_generator.GetPrimaryResource(self.methods, args).method
        poller = AsyncOperationPoller(self.spec, resource_ref if extract_resource_result else None, args, operation_collection, method)
        if poller.IsDone(operation):
            return poller.GetResult(operation)
        return self._WaitForOperationWithPoller(poller, operation_ref, args=args)

    def _WaitForOperationWithPoller(self, poller, operation_ref, args=None):
        progress_string = self._Format('Waiting for operation [{{{}}}] to complete'.format(yaml_command_schema_util.REL_NAME_FORMAT_KEY), operation_ref, self._GetDisplayResourceType(args))
        display_name = self._GetDisplayName(poller.resource_ref, args) if args else None
        return waiter.WaitFor(poller, operation_ref, self._Format(progress_string, poller.resource_ref, self._GetDisplayResourceType(args), display_name))

    def _FindPopulatedAttribute(self, obj, attributes):
        """Searches the given object for an attribute that is non-None.

    This digs into the object search for the given attributes. If any attribute
    along the way is a list, it will search for sub-attributes in each item
    of that list. The first match is returned.

    Args:
      obj: The object to search
      attributes: [str], A sequence of attributes to use to dig into the
        resource.

    Returns:
      The first matching instance of the attribute that is non-None, or None
      if one could nto be found.
    """
        if not attributes:
            return obj
        attr = attributes[0]
        try:
            obj = getattr(obj, attr)
        except AttributeError:
            return None
        if isinstance(obj, list):
            for x in obj:
                obj = self._FindPopulatedAttribute(x, attributes[1:])
                if obj:
                    return obj
        return self._FindPopulatedAttribute(obj, attributes[1:])

    def _GetExistingResource(self, args):
        from googlecloudsdk.command_lib.util.apis import arg_marshalling
        get_methods = self._GetMethods('get')
        specified = self.arg_generator.GetPrimaryResource(get_methods, args)
        primary_resource_arg = specified.primary_resource
        params = [primary_resource_arg] if primary_resource_arg else []
        get_arg_generator = arg_marshalling.DeclarativeArgumentGenerator(params)
        get_method = specified.method
        return get_method.Call(get_arg_generator.CreateRequest(args, get_method))

    def _ConfigureCommand(self, command):
        """Configures top level attributes of the generated command.

    Args:
      command: The command being generated.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        if self.spec.hidden:
            command = base.Hidden(command)
        if self.spec.universe_compatible is not None:
            if self.spec.universe_compatible:
                command = base.UniverseCompatible(command)
            else:
                command = base.DefaultUniverseOnly(command)
        if self.spec.release_tracks:
            command = base.ReleaseTracks(*self.spec.release_tracks)(command)
        if self.spec.deprecated_data:
            command = base.Deprecate(**self.spec.deprecated_data)(command)
        if not hasattr(command, 'detailed_help'):
            key_map = {'description': 'DESCRIPTION', 'examples': 'EXAMPLES'}
            command.detailed_help = {key_map.get(k, k): v for k, v in self.spec.help_text.items()}
        if self.has_request_method:
            api_names = set((f'{method.collection.api_name}/{method.collection.api_version}' for method in self.methods))
            doc_urls = set((method.collection.docs_url for method in self.methods))
            api_name_str = ', '.join(api_names)
            doc_url_str = ', '.join(doc_urls)
            if len(api_names) > 1:
                api_info = f'This command uses *{api_name_str}* APIs. The full documentation for these APIs can be found at: {doc_url_str}'
            else:
                api_info = f'This command uses the *{api_name_str}* API. The full documentation for this API can be found at: {doc_url_str}'
            command.detailed_help['API REFERENCE'] = api_info
        return command

    @abc.abstractmethod
    def _Generate(self):
        pass

    def Generate(self):
        command = self._Generate()
        self._ConfigureCommand(command)
        return command