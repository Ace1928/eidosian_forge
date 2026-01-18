import re
import sys
from copy import copy
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterable, Iterator, List, Optional,
from docutils import nodes
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives
from docutils.statemachine import StringList
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, ObjType
from sphinx.locale import _, __
from sphinx.roles import EmphasizedLiteral, XRefRole
from sphinx.util import docname_join, logging, ws_re
from sphinx.util.docutils import SphinxDirective
from sphinx.util.nodes import clean_astext, make_id, make_refnode
from sphinx.util.typing import OptionSpec, RoleFunction
class Cmdoption(ObjectDescription[str]):
    """
    Description of a command-line option (.. option).
    """

    def handle_signature(self, sig: str, signode: desc_signature) -> str:
        """Transform an option description into RST nodes."""
        count = 0
        firstname = ''
        for potential_option in sig.split(', '):
            potential_option = potential_option.strip()
            m = option_desc_re.match(potential_option)
            if not m:
                logger.warning(__('Malformed option description %r, should look like "opt", "-opt args", "--opt args", "/opt args" or "+opt args"'), potential_option, location=signode)
                continue
            optname, args = m.groups()
            if optname[-1] == '[' and args[-1] == ']':
                optname = optname[:-1]
                args = '[' + args
            if count:
                if self.env.config.option_emphasise_placeholders:
                    signode += addnodes.desc_sig_punctuation(',', ',')
                    signode += addnodes.desc_sig_space()
                else:
                    signode += addnodes.desc_addname(', ', ', ')
            signode += addnodes.desc_name(optname, optname)
            if self.env.config.option_emphasise_placeholders:
                add_end_bracket = False
                if args:
                    if args[0] == '[' and args[-1] == ']':
                        add_end_bracket = True
                        signode += addnodes.desc_sig_punctuation('[', '[')
                        args = args[1:-1]
                    elif args[0] == ' ':
                        signode += addnodes.desc_sig_space()
                        args = args.strip()
                    elif args[0] == '=':
                        signode += addnodes.desc_sig_punctuation('=', '=')
                        args = args[1:]
                    for part in samp_role.parse(args):
                        if isinstance(part, nodes.Text):
                            signode += nodes.Text(part.astext())
                        else:
                            signode += part
                if add_end_bracket:
                    signode += addnodes.desc_sig_punctuation(']', ']')
            else:
                signode += addnodes.desc_addname(args, args)
            if not count:
                firstname = optname
                signode['allnames'] = [optname]
            else:
                signode['allnames'].append(optname)
            count += 1
        if not firstname:
            raise ValueError
        return firstname

    def add_target_and_index(self, firstname: str, sig: str, signode: desc_signature) -> None:
        currprogram = self.env.ref_context.get('std:program')
        for optname in signode.get('allnames', []):
            prefixes = ['cmdoption']
            if currprogram:
                prefixes.append(currprogram)
            if not optname.startswith(('-', '/')):
                prefixes.append('arg')
            prefix = '-'.join(prefixes)
            node_id = make_id(self.env, self.state.document, prefix, optname)
            signode['ids'].append(node_id)
            old_node_id = self.make_old_id(prefix, optname)
            if old_node_id not in self.state.document.ids and old_node_id not in signode['ids']:
                signode['ids'].append(old_node_id)
        self.state.document.note_explicit_target(signode)
        domain = cast(StandardDomain, self.env.get_domain('std'))
        for optname in signode.get('allnames', []):
            domain.add_program_option(currprogram, optname, self.env.docname, signode['ids'][0])
        if currprogram:
            descr = _('%s command line option') % currprogram
        else:
            descr = _('command line option')
        for option in signode.get('allnames', []):
            entry = '; '.join([descr, option])
            self.indexnode['entries'].append(('pair', entry, signode['ids'][0], '', None))

    def make_old_id(self, prefix: str, optname: str) -> str:
        """Generate old styled node_id for cmdoption.

        .. note:: Old Styled node_id was used until Sphinx-3.0.
                  This will be removed in Sphinx-5.0.
        """
        return nodes.make_id(prefix + '-' + optname)