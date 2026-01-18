import os
import six
import sys
from pyparsing import (alphanums, Empty, Group, locatedExpr,
from . import console
from . import log
from . import prefs
from .node import ConfigNode, ExecutionError
import signal
def _complete_token_path(self, text):
    """
        Completes a partial path token.
        @param text: Current text being typed by the user.
        @type text: str
        @return: Possible completions for the token.
        @rtype: list of str
        """
    completions = []
    if text.endswith('.'):
        text = text + '/'
    basedir, slash, partial_name = text.rpartition('/')
    self.log.debug('Got basedir=%s, partial_name=%s' % (basedir, partial_name))
    basedir = basedir + slash
    target = self._current_node.get_node(basedir)
    names = [child.name for child in target.children]
    if names and partial_name in ['', '*']:
        if len(names) > 1:
            completions.append('%s* ' % basedir)
    for name in names:
        num_matches = 0
        if name.startswith(partial_name):
            num_matches += 1
            if num_matches == 1:
                completions.append('%s%s/' % (basedir, name))
            else:
                completions.append('%s%s' % (basedir, name))
    bookmarks = ['@' + bookmark for bookmark in self.prefs['bookmarks'] if bookmark.startswith('%s' % text.lstrip('@'))]
    self.log.debug('Found bookmarks %s.' % str(bookmarks))
    if bookmarks:
        completions.extend(bookmarks)
    if len(completions) == 1:
        self.log.debug('One completion left.')
        if not completions[0].endswith('* '):
            if not self._current_node.get_node(completions[0]).children:
                completions[0] = completions[0].rstrip('/') + ' '
    self._current_token = self.con.render_text('path', self.prefs['color_path'])
    return completions