from __future__ import absolute_import, division, print_function
from ansible.module_utils._text import to_native
from ansible_collections.ansible.utils.plugins.plugin_utils.base.cli_parser import CliParserBase
from ansible_collections.ansible.netcommon.plugins.module_utils.cli_parser.cli_parsertemplate import (
class CliParser(CliParserBase):
    """The native parser class
    Convert raw text to structured data using the resource module parser
    """
    DEFAULT_TEMPLATE_EXTENSION = 'yaml'
    PROVIDE_TEMPLATE_CONTENTS = True

    def parse(self, *_args, **kwargs):
        """Std entry point for a cli_parse parse execution

        :return: Errors or parsed text as structured data
        :rtype: dict

        :example:

        The parse function of a parser should return a dict:
        {"errors": [a list of errors]}
        or
        {"parsed": obj}
        """
        template_contents = kwargs['template_contents']
        parser = CliParserTemplate(lines=self._task_args.get('text', '').splitlines())
        try:
            template_obj = list(eval(template_contents))
        except Exception as exc:
            return {'errors': [to_native(exc)]}
        try:
            parser.PARSERS = template_obj
            out = {'parsed': parser.parse()}
            print(out)
            return out
        except Exception as exc:
            msg = 'Native parser returned an error while parsing. Error: {err}'
            return {'errors': [msg.format(err=to_native(exc))]}