import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
@register_script('convo_render', hidden=True)
class RenderConversation(ParlaiScript):

    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return render_convo(self.opt)