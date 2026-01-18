from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from prompt_toolkit import styles
from prompt_toolkit import token
def Color(foreground=None, background=None, bold=False):
    components = []
    if foreground:
        components.append(foreground)
    if background:
        components.append('bg:' + background)
    if bold:
        components.append('bold')
    return ' '.join(components)