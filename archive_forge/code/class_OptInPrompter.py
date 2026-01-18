from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import time
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import files as file_utils
import six
class OptInPrompter(BasePrompter):
    """Prompter to opt-in in data usage."""
    PROMPT_INTERVAL = 86400 * 30 * 2
    MESSAGE = "To help improve the quality of this product, we collect anonymized usage data and anonymized stacktraces when crashes are encountered; additional information is available at <https://cloud.google.com/sdk/usage-statistics>. This data is handled in accordance with our privacy policy <https://cloud.google.com/terms/cloud-privacy-notice>. You may choose to opt in this collection now (by choosing 'Y' at the below prompt), or at any time in the future by running the following command:\n\n    gcloud config set disable_usage_reporting false\n"

    def __init__(self):
        self.record = OptInPromptRecord()

    def Prompt(self):
        """Asks users to opt-in data usage report."""
        if not properties.IsDefaultUniverse():
            return
        if not self.record.CacheFileExists():
            with self.record as pr:
                pr.last_prompt_time = 0
        if self.ShouldPrompt():
            answer = console_io.PromptContinue(message=self.MESSAGE, prompt_string='Do you want to opt-in', default=False, throw_if_unattended=False, cancel_on_no=False)
            if answer:
                properties.PersistProperty(properties.VALUES.core.disable_usage_reporting, 'False')
            with self.record as pr:
                pr.last_prompt_time = time.time()

    def ShouldPrompt(self):
        """Checks whether to prompt or not."""
        if not (log.out.isatty() and log.err.isatty()):
            return False
        last_prompt_time = self.record.last_prompt_time
        now = time.time()
        if last_prompt_time and now - last_prompt_time < self.PROMPT_INTERVAL:
            return False
        return True