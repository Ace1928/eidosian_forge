import json
import os
from parlai.core.teachers import FixedDialogTeacher
from .build import build
from parlai.tasks.multinli.agents import convert_to_dialogData
class ExtrasTeacher(DialogueNliTeacher):

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared, extras=True)