from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.survey import concord_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.survey import survey
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
def _GetAnswerToQuestion(question):
    """Prompts user for the answer to the question."""
    prompt_msg = question.instruction
    while True:
        answer = console_io.PromptResponse(prompt_msg)
        if answer == survey.Survey.ControlOperation.SKIP_QUESTION.value:
            return survey.Survey.ControlOperation.SKIP_QUESTION
        elif answer == survey.Survey.ControlOperation.EXIT_SURVEY.value:
            return survey.Survey.ControlOperation.EXIT_SURVEY
        elif question.AcceptAnswer(answer):
            return answer
        else:
            prompt_msg = question.instruction_on_rejection