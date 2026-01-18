from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core.console import console_io
def ResourceFromFreeformPrompt(name, long_name, list_func):
    """Prompts the user to select a resource.

  Args:
    name: the name of the resource. For example, "environment" or "developer".
    long_name: a longer form of `name` which the user will see in prompts.
      Should explain the context in which the resource will be used. For
      example, "the environment to be updated".
    list_func: a function that returns the names of existing resources.

  Returns:
    The resource's identifier if successful, or None if not.
  """
    resource_list = []
    try:
        resource_list = list_func()
    except errors.RequestError:
        pass
    entity_names = resource_args.ENTITIES[name]
    if resource_list:
        enter_manually = '(some other %s)' % entity_names.docs_name
        choice = console_io.PromptChoice(resource_list + [enter_manually], prompt_string='Select %s:' % long_name)
        if choice < len(resource_list):
            return resource_list[choice]
    valid_pattern = resource_args.ValidPatternForEntity(name)
    validator = lambda response: valid_pattern.search(response) is not None
    error_str = "Doesn't match the expected format of a " + entity_names.docs_name
    prompt_message = 'Enter %s: ' % long_name
    return console_io.PromptWithValidator(validator, error_str, prompt_message)