import logging
import stevedore
from cliff import command
class CompleteBash(CompleteShellBase):
    """completion for bash
    """

    def __init__(self, name, output):
        super(CompleteBash, self).__init__(name, output)

    def get_header(self):
        return '_' + self.escaped_name + '()\n{\n  local cur prev words\n  COMPREPLY=()\n  _get_comp_words_by_ref -n : cur prev words\n\n  # Command data:\n'

    def get_trailer(self):
        return '\n  dash=-\n  underscore=_\n  cmd=""\n  words[0]=""\n  completed="${cmds}"\n  for var in "${words[@]:1}"\n  do\n    if [[ ${var} == -* ]] ; then\n      break\n    fi\n    if [ -z "${cmd}" ] ; then\n      proposed="${var}"\n    else\n      proposed="${cmd}_${var}"\n    fi\n    local i="cmds_${proposed}"\n    i=${i//$dash/$underscore}\n    local comp="${!i}"\n    if [ -z "${comp}" ] ; then\n      break\n    fi\n    if [[ ${comp} == -* ]] ; then\n      if [[ ${cur} != -* ]] ; then\n        completed=""\n        break\n      fi\n    fi\n    cmd="${proposed}"\n    completed="${comp}"\n  done\n\n  if [ -z "${completed}" ] ; then\n    COMPREPLY=( $( compgen -f -- "$cur" ) $( compgen -d -- "$cur" ) )\n  else\n    COMPREPLY=( $(compgen -W "${completed}" -- ${cur}) )\n  fi\n  return 0\n}\ncomplete -F _' + self.escaped_name + ' ' + self.name + '\n'