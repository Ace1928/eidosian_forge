import re
import sys
import breezy
from ... import cmdline, commands, config, help_topics, option, plugin
class BashCodeGen:
    """Generate a bash script for given completion data."""

    def __init__(self, data, function_name='_brz', debug=False):
        self.data = data
        self.function_name = function_name
        self.debug = debug

    def script(self):
        return '# Programmable completion for the Breezy brz command under bash.\n# Known to work with bash 2.05a as well as bash 4.1.2, and probably\n# all versions in between as well.\n\n# Based originally on the svn bash completition script.\n# Customized by Sven Wilhelm/Icecrash.com\n# Adjusted for automatic generation by Martin von Gagern\n\n# Generated using the bash_completion plugin.\n# See https://launchpad.net/bzr-bash-completion for details.\n\n# Commands and options of brz {brz_version}\n\nshopt -s progcomp\n{function}\ncomplete -F {function_name} -o default brz\n'.format(function_name=self.function_name, function=self.function(), brz_version=self.brz_version())

    def function(self):
        return '%(function_name)s ()\n{\n    local cur cmds cmdIdx cmd cmdOpts fixedWords i globalOpts\n    local curOpt optEnums\n    local IFS=$\' \\n\'\n\n    COMPREPLY=()\n    cur=${COMP_WORDS[COMP_CWORD]}\n\n    cmds=\'%(cmds)s\'\n    globalOpts=( %(global_options)s )\n\n    # do ordinary expansion if we are anywhere after a -- argument\n    for ((i = 1; i < COMP_CWORD; ++i)); do\n        [[ ${COMP_WORDS[i]} == "--" ]] && return 0\n    done\n\n    # find the command; it\'s the first word not starting in -\n    cmd=\n    for ((cmdIdx = 1; cmdIdx < ${#COMP_WORDS[@]}; ++cmdIdx)); do\n        if [[ ${COMP_WORDS[cmdIdx]} != -* ]]; then\n            cmd=${COMP_WORDS[cmdIdx]}\n            break\n        fi\n    done\n\n    # complete command name if we are not already past the command\n    if [[ $COMP_CWORD -le cmdIdx ]]; then\n        COMPREPLY=( $( compgen -W "$cmds ${globalOpts[*]}" -- $cur ) )\n        return 0\n    fi\n\n    # find the option for which we want to complete a value\n    curOpt=\n    if [[ $cur != -* ]] && [[ $COMP_CWORD -gt 1 ]]; then\n        curOpt=${COMP_WORDS[COMP_CWORD - 1]}\n        if [[ $curOpt == = ]]; then\n            curOpt=${COMP_WORDS[COMP_CWORD - 2]}\n        elif [[ $cur == : ]]; then\n            cur=\n            curOpt="$curOpt:"\n        elif [[ $curOpt == : ]]; then\n            curOpt=${COMP_WORDS[COMP_CWORD - 2]}:\n        fi\n    fi\n%(debug)s\n    cmdOpts=( )\n    optEnums=( )\n    fixedWords=( )\n    case $cmd in\n%(cases)s    *)\n        cmdOpts=(--help -h)\n        ;;\n    esac\n\n    IFS=$\'\\n\'\n    if [[ ${#fixedWords[@]} -eq 0 ]] && [[ ${#optEnums[@]} -eq 0 ]] && [[ $cur != -* ]]; then\n        case $curOpt in\n            tag:|*..tag:)\n                fixedWords=( $(brz tags 2>/dev/null | sed \'s/  *[^ ]*$//; s/ /\\\\\\\\ /g;\') )\n                ;;\n        esac\n        case $cur in\n            [\\"\\\']tag:*)\n                fixedWords=( $(brz tags 2>/dev/null | sed \'s/  *[^ ]*$//; s/^/tag:/\') )\n                ;;\n            [\\"\\\']*..tag:*)\n                fixedWords=( $(brz tags 2>/dev/null | sed \'s/  *[^ ]*$//\') )\n                fixedWords=( $(for i in "${fixedWords[@]}"; do echo "${cur%%..tag:*}..tag:${i}"; done) )\n                ;;\n        esac\n    elif [[ $cur == = ]] && [[ ${#optEnums[@]} -gt 0 ]]; then\n        # complete directly after "--option=", list all enum values\n        COMPREPLY=( "${optEnums[@]}" )\n        return 0\n    else\n        fixedWords=( "${cmdOpts[@]}"\n                     "${globalOpts[@]}"\n                     "${optEnums[@]}"\n                     "${fixedWords[@]}" )\n    fi\n\n    if [[ ${#fixedWords[@]} -gt 0 ]]; then\n        COMPREPLY=( $( compgen -W "${fixedWords[*]}" -- $cur ) )\n    fi\n\n    return 0\n}\n' % {'cmds': self.command_names(), 'function_name': self.function_name, 'cases': self.command_cases(), 'global_options': self.global_options(), 'debug': self.debug_output()}

    def command_names(self):
        return ' '.join(self.data.all_command_aliases())

    def debug_output(self):
        if not self.debug:
            return ''
        else:
            return '\n    # Debugging code enabled using the --debug command line switch.\n    # Will dump some variables to the top portion of the terminal.\n    echo -ne \'\\e[s\\e[H\'\n    for (( i=0; i < ${#COMP_WORDS[@]}; ++i)); do\n        echo "\\$COMP_WORDS[$i]=\'${COMP_WORDS[i]}\'"$\'\\e[K\'\n    done\n    for i in COMP_CWORD COMP_LINE COMP_POINT COMP_TYPE COMP_KEY cur curOpt; do\n        echo "\\$${i}=\\"${!i}\\""$\'\\e[K\'\n    done\n    echo -ne \'---\\e[K\\e[u\'\n'

    def brz_version(self):
        brz_version = breezy.version_string
        if not self.data.plugins:
            brz_version += '.'
        else:
            brz_version += ' and the following plugins:'
            for name, plugin in sorted(self.data.plugins.items()):
                brz_version += '\n# %s' % plugin
        return brz_version

    def global_options(self):
        return ' '.join(sorted(self.data.global_options))

    def command_cases(self):
        cases = ''
        for command in self.data.commands:
            cases += self.command_case(command)
        return cases

    def command_case(self, command):
        case = '\t%s)\n' % '|'.join(command.aliases)
        if command.plugin:
            case += '\t\t# plugin "%s"\n' % command.plugin
        options = []
        enums = []
        for option in command.options:
            for message in option.error_messages:
                case += '\t\t# %s\n' % message
            if option.registry_keys:
                for key in option.registry_keys:
                    options.append('{}={}'.format(option, key))
                enums.append('%s) optEnums=( %s ) ;;' % (option, ' '.join(option.registry_keys)))
            else:
                options.append(str(option))
        case += '\t\tcmdOpts=( %s )\n' % ' '.join(options)
        if command.fixed_words:
            fixed_words = command.fixed_words
            if isinstance(fixed_words, list):
                fixed_words = '( %s )' + ' '.join(fixed_words)
            case += '\t\tfixedWords=%s\n' % fixed_words
        if enums:
            case += '\t\tcase $curOpt in\n\t\t\t'
            case += '\n\t\t\t'.join(enums)
            case += '\n\t\tesac\n'
        case += '\t\t;;\n'
        return case