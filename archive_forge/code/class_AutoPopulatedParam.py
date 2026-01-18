import re
from collections import namedtuple
class AutoPopulatedParam:

    def __init__(self, name, param_description=None):
        self.name = name
        self.param_description = param_description
        if param_description is None:
            self.param_description = 'Please note that this parameter is automatically populated if it is not provided. Including this parameter is not required\n'

    def document_auto_populated_param(self, event_name, section, **kwargs):
        """Documents auto populated parameters

        It will remove any required marks for the parameter, remove the
        parameter from the example, and add a snippet about the parameter
        being autopopulated in the description.
        """
        if event_name.startswith('docs.request-params'):
            if self.name in section.available_sections:
                section = section.get_section(self.name)
                if 'is-required' in section.available_sections:
                    section.delete_section('is-required')
                description_section = section.get_section('param-documentation')
                description_section.writeln(self.param_description)
        elif event_name.startswith('docs.request-example'):
            section = section.get_section('structure-value')
            if self.name in section.available_sections:
                section.delete_section(self.name)