from __future__ import annotations
import logging
from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_community.tools.edenai.edenai_base_tool import EdenaiTool
class EdenAiTextModerationTool(EdenaiTool):
    """Tool that queries the Eden AI Explicit text detection.

    for api reference check edenai documentation:
    https://docs.edenai.co/reference/image_explicit_content_create.

    To use, you should have
    the environment variable ``EDENAI_API_KEY`` set with your API token.
    You can find your token here: https://app.edenai.run/admin/account/settings

    """
    name = 'edenai_explicit_content_detection_text'
    description = "A wrapper around edenai Services explicit content detection for text. Useful for when you have to scan text for offensive, \n        sexually explicit or suggestive content,\n        it checks also if there is any content of self-harm,\n        violence, racist or hate speech.the structure of the output is : \n        'the type of the explicit content : the likelihood of it being explicit'\n        the likelihood is a number \n        between 1 and 5, 1 being the lowest and 5 the highest.\n        something is explicit if the likelihood is equal or higher than 3.\n        for example : \n        nsfw_likelihood: 1\n        this is not explicit.\n        for example : \n        nsfw_likelihood: 3\n        this is explicit.\n        Input should be a string."
    language: str
    feature: str = 'text'
    subfeature: str = 'moderation'

    def _parse_response(self, response: list) -> str:
        formatted_result = []
        for result in response:
            if 'nsfw_likelihood' in result.keys():
                formatted_result.append('nsfw_likelihood: ' + str(result['nsfw_likelihood']))
            for label, likelihood in zip(result['label'], result['likelihood']):
                formatted_result.append(f'"{label}": {str(likelihood)}')
        return '\n'.join(formatted_result)

    def _run(self, query: str, run_manager: Optional[CallbackManagerForToolRun]=None) -> str:
        """Use the tool."""
        query_params = {'text': query, 'language': self.language}
        return self._call_eden_ai(query_params)