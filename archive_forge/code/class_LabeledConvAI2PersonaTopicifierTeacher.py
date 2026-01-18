import os
from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.style_gen.build import (
class LabeledConvAI2PersonaTopicifierTeacher(ParlAIDialogTeacher):
    """
    Teacher for blended_skill_talk:ConvAI2PersonaTopicifier, with Image-Chat
    personalities added to examples.
    """

    def __init__(self, opt, shared=None):
        opt['parlaidialogteacher_datafile'] = get_style_labeled_data_path(opt=opt, base_task='blended_skill_talk:ConvAI2PersonaTopicifierTeacher')
        super().__init__(opt, shared=shared)