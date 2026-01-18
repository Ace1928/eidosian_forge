import unittest
from unittest.mock import patch
import torch
from transformers import AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead, TextEnvironment, TextHistory
class TextHistoryTest(unittest.TestCase):

    def test_text_history_init(self):
        text = 'Hello there!'
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        assert history.text == text
        assert torch.equal(history.tokens, tokens)
        assert torch.equal(history.token_masks, torch.zeros_like(tokens))
        history = TextHistory(text, tokens, system=False)
        assert torch.equal(history.token_masks, torch.ones_like(tokens))

    def test_text_history_append_segment(self):
        text = 'Hello there!'
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]), system=False)
        assert history.text == text + 'General Kenobi!'
        assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6]))
        assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1]))
        history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]))
        assert history.text == text + 'General Kenobi!' + 'You are a bold one!'
        assert torch.equal(history.tokens, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        assert torch.equal(history.token_masks, torch.tensor([0, 0, 0, 1, 1, 1, 0, 0, 0]))

    def test_text_history_complete(self):
        text = 'Hello there!'
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.complete()
        assert history.completed
        assert not history.truncated
        history.complete(truncated=True)
        assert history.completed
        assert history.truncated

    def test_text_history_last_segment(self):
        text = 'Hello there!'
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]))
        history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]))
        assert history.last_text_segment == 'You are a bold one!'

    def test_text_history_split_query_response(self):
        text = 'Hello there!'
        tokens = torch.tensor([1, 2, 3])
        history = TextHistory(text, tokens)
        history.append_segment('General Kenobi!', torch.tensor([4, 5, 6]), system=False)
        history.append_segment('You are a bold one!', torch.tensor([7, 8, 9]), system=True)
        query, response, mask = history.split_query_response_tokens()
        assert torch.equal(query, torch.tensor([1, 2, 3]))
        assert torch.equal(response, torch.tensor([4, 5, 6, 7, 8, 9]))
        assert torch.equal(mask, torch.tensor([1, 1, 1, 0, 0, 0]))