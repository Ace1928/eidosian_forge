import os
from unittest import TestCase
from llama.tokenizer import ChatFormat, Tokenizer
class TokenizerTests(TestCase):

    def setUp(self):
        self.tokenizer = Tokenizer(os.environ['TOKENIZER_PATH'])
        self.format = ChatFormat(self.tokenizer)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.special_tokens['<|begin_of_text|>'], 128000)

    def test_encode(self):
        self.assertEqual(self.tokenizer.encode('This is a test sentence.', bos=True, eos=True), [128000, 2028, 374, 264, 1296, 11914, 13, 128001])

    def test_decode(self):
        self.assertEqual(self.tokenizer.decode([128000, 2028, 374, 264, 1296, 11914, 13, 128001]), '<|begin_of_text|>This is a test sentence.<|end_of_text|>')

    def test_encode_message(self):
        message = {'role': 'user', 'content': 'This is a test sentence.'}
        self.assertEqual(self.format.encode_message(message), [128006, 882, 128007, 271, 2028, 374, 264, 1296, 11914, 13, 128009])

    def test_encode_dialog(self):
        dialog = [{'role': 'system', 'content': 'This is a test sentence.'}, {'role': 'user', 'content': 'This is a response.'}]
        self.assertEqual(self.format.encode_dialog_prompt(dialog), [128000, 128006, 9125, 128007, 271, 2028, 374, 264, 1296, 11914, 13, 128009, 128006, 882, 128007, 271, 2028, 374, 264, 2077, 13, 128009, 128006, 78191, 128007, 271])