import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
class TextHistory:
    """The TextHistory class keeps track of the history of an interaction between the language model and the environment."""

    def __init__(self, text, tokens, system=True):
        """
        Initialize TextHistory.

        args:
            text (`str`): The text of the first segment.
            tokens (`torch.LongTensor`): The tokens of the first segment.
            system (`bool`, *optional*): Whether the first segment is a system or user segment.
        """
        self.system_spans = []
        self.text_spans = []
        self.token_spans = []
        self.token_masks = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.text = ''
        self.tokens = torch.tensor([], dtype=torch.long).to(tokens.device)
        self.completed = False
        self.truncated = False
        self.reward = 0.0
        self.prompt_color = 'black on grey85'
        self.system_color = 'black on cyan3'
        self.model_color = 'black on deep_sky_blue1'
        self.reward_color = 'black on plum1'
        self.append_segment(text, tokens, system=system)

    def append_segment(self, text, tokens, system=True):
        """
        Append a new segment to the history.

        args:
            text (`str`): The text of the new segment.
            tokens (`torch.LongTensor`): The tokens of the new segment.
            system (`bool`, *optional*): Whether the new segment is a system or user segment.
        """
        if len(text) == 0 or len(tokens) == 0:
            raise ValueError("Can't append empty text or token list to history.")
        original_text_length = len(self.text)
        self.text += text
        self.text_spans.append((original_text_length, len(self.text)))
        self.system_spans.append(system)
        original_token_length = len(self.tokens)
        self.tokens = torch.cat((self.tokens, tokens))
        if system:
            self.token_masks = torch.cat((self.token_masks, torch.zeros_like(tokens)))
        else:
            self.token_masks = torch.cat((self.token_masks, torch.ones_like(tokens)))
        self.token_spans.append((original_token_length, len(self.tokens)))

    def complete(self, truncated=False):
        """
        Mark the history as completed.
        """
        self.completed = True
        self.truncated = truncated

    @property
    def last_text_segment(self):
        """
        Get the last text segment.
        """
        start, end = self.text_spans[-1]
        return self.text[start:end]

    def split_query_response_tokens(self):
        """
        Split the tokens into query and response tokens.
        """
        split_index = self.token_spans[0][1]
        query = self.tokens[:split_index]
        response = self.tokens[split_index:]
        mask = self.token_masks[split_index:]
        return (query, response, mask)

    def show_text(self, show_legend=False):
        """
        Print the text history.
        """
        if not is_rich_available():
            warnings.warn('install rich to display text')
            return
        text = Text(self.text)
        text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])
        for i, (start, end) in enumerate(self.text_spans[1:]):
            if self.system_spans[i + 1]:
                text.stylize(self.system_color, start, end)
            else:
                text.stylize(self.model_color, start, end)
        text.append(f'\n\nReward: {self.reward}', style=self.reward_color)
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_tokens(self, tokenizer, show_legend=False):
        """
        Print the history tokens.
        """
        if not is_rich_available():
            warnings.warn('install rich to display tokens')
            return
        text = Text()
        prompt_end = self.token_spans[0][1]
        for i, (token, mask) in enumerate(zip(self.tokens, self.token_masks)):
            if i < prompt_end:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.prompt_color)
                text.append(' ')
            elif mask == 0:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.system_color)
                text.append(' ')
            else:
                text.append(tokenizer.convert_ids_to_tokens(token.item()), style=self.model_color)
                text.append(' ')
        text.append(f'\n\nReward: {self.reward}', style=self.reward_color)
        print(text)
        if show_legend:
            self.show_colour_legend()

    def show_colour_legend(self):
        """
        Print the colour legend.
        """
        if not is_rich_available():
            warnings.warn('install rich to display colour legend')
            return
        text = Text('\n\n(Colour Legend: ')
        text.append('Prompt', style=self.prompt_color)
        text.append('|')
        text.append('System', style=self.system_color)
        text.append('|')
        text.append('Model', style=self.model_color)
        text.append('|')
        text.append('Reward', style=self.reward_color)
        text.append(')')
        print(text)