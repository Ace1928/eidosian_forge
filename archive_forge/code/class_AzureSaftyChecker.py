import os
import torch
import warnings
from typing import List
from string import Template
from enum import Enum
class AzureSaftyChecker(object):

    def __init__(self):
        try:
            from azure.ai.contentsafety import ContentSafetyClient
            from azure.core.credentials import AzureKeyCredential
            key = os.environ['CONTENT_SAFETY_KEY']
            endpoint = os.environ['CONTENT_SAFETY_ENDPOINT']
        except ImportError:
            raise Exception('Could not import required package azure-ai-contentsafety. Install with: pip install azure-ai-contentsafety')
        except KeyError:
            raise Exception('Environment variables not set. Please set CONTENT_SAFETY_KEY and CONTENT_SAFETY_ENDPOINT.')
        self.client = ContentSafetyClient(endpoint, AzureKeyCredential(key))

    def __call__(self, output_text, **kwargs):
        from azure.core.exceptions import HttpResponseError
        from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
        print(len(output_text))
        if len(output_text) > 1000:
            raise Exception('Input length to safety check is too long (>1000).')
        categories = [TextCategory.VIOLENCE, TextCategory.SELF_HARM, TextCategory.SEXUAL, TextCategory.HATE]
        request = AnalyzeTextOptions(text=output_text, categories=categories)
        try:
            response = self.client.analyze_text(request)
        except HttpResponseError as e:
            print('Analyze text failed.')
            if e.error:
                print(f'Error code: {e.error.code}')
                print(f'Error message: {e.error.message}')
                raise
            print(e)
            raise e
        levels = {0: 'Safe', 2: 'Low', 4: 'Medium', 6: 'High'}
        severities = [getattr(response, c.name.lower() + '_result').severity for c in categories]
        DEFAULT_LEVELS = [0, 0, 0, 0]
        is_safe = all([s <= l for s, l in zip(severities, DEFAULT_LEVELS)])
        report = ''
        if not is_safe:
            report = '|' + '|'.join((f'{c.name:^10}' for c in categories)) + '|\n'
            report += '|' + '|'.join((f'{levels[s]:^10}' for s in severities)) + '|\n'
        return ('Azure Content Saftey API', is_safe, report)