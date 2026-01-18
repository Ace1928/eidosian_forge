from typing import List, Tuple
import pytest
from nltk.tokenize import (
class TestTokenize:

    def test_tweet_tokenizer(self):
        """
        Test TweetTokenizer using words with special and accented characters.
        """
        tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
        s9 = "@myke: Let's test these words: resumé España München français"
        tokens = tokenizer.tokenize(s9)
        expected = [':', "Let's", 'test', 'these', 'words', ':', 'resumé', 'España', 'München', 'français']
        assert tokens == expected

    @pytest.mark.parametrize('test_input, expecteds', [('My text 0106404243030 is great text', (['My', 'text', '01064042430', '30', 'is', 'great', 'text'], ['My', 'text', '0106404243030', 'is', 'great', 'text'])), ('My ticket id is 1234543124123', (['My', 'ticket', 'id', 'is', '12345431241', '23'], ['My', 'ticket', 'id', 'is', '1234543124123'])), ('@remy: This is waaaaayyyy too much for you!!!!!! 01064042430', ([':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!', '01064042430'], [':', 'This', 'is', 'waaayyy', 'too', 'much', 'for', 'you', '!', '!', '!', '01064042430'])), ("My number is 06-46124080, except it's not.", (['My', 'number', 'is', '06-46124080', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '06-46124080', ',', 'except', "it's", 'not', '.'])), ("My number is 601-984-4813, except it's not.", (['My', 'number', 'is', '601-984-4813', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '601-984-', '4813', ',', 'except', "it's", 'not', '.'])), ("My number is (393)  928 -3010, except it's not.", (['My', 'number', 'is', '(393)  928 -3010', ',', 'except', "it's", 'not', '.'], ['My', 'number', 'is', '(', '393', ')', '928', '-', '3010', ',', 'except', "it's", 'not', '.'])), ('The product identification number is 48103284512.', (['The', 'product', 'identification', 'number', 'is', '4810328451', '2', '.'], ['The', 'product', 'identification', 'number', 'is', '48103284512', '.'])), ('My favourite substraction is 240 - 1353.', (['My', 'favourite', 'substraction', 'is', '240 - 1353', '.'], ['My', 'favourite', 'substraction', 'is', '240', '-', '1353', '.']))])
    def test_tweet_tokenizer_expanded(self, test_input: str, expecteds: Tuple[List[str], List[str]]):
        """
        Test `match_phone_numbers` in TweetTokenizer.

        Note that TweetTokenizer is also passed the following for these tests:
            * strip_handles=True
            * reduce_len=True

        :param test_input: The input string to tokenize using TweetTokenizer.
        :type test_input: str
        :param expecteds: A 2-tuple of tokenized sentences. The first of the two
            tokenized is the expected output of tokenization with `match_phone_numbers=True`.
            The second of the two tokenized lists is the expected output of tokenization
            with `match_phone_numbers=False`.
        :type expecteds: Tuple[List[str], List[str]]
        """
        for match_phone_numbers, expected in zip([True, False], expecteds):
            tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True, match_phone_numbers=match_phone_numbers)
            predicted = tokenizer.tokenize(test_input)
            assert predicted == expected

    def test_sonority_sequencing_syllable_tokenizer(self):
        """
        Test SyllableTokenizer tokenizer.
        """
        tokenizer = SyllableTokenizer()
        tokens = tokenizer.tokenize('justification')
        assert tokens == ['jus', 'ti', 'fi', 'ca', 'tion']

    def test_syllable_tokenizer_numbers(self):
        """
        Test SyllableTokenizer tokenizer.
        """
        tokenizer = SyllableTokenizer()
        text = '9' * 10000
        tokens = tokenizer.tokenize(text)
        assert tokens == [text]

    def test_legality_principle_syllable_tokenizer(self):
        """
        Test LegalitySyllableTokenizer tokenizer.
        """
        from nltk.corpus import words
        test_word = 'wonderful'
        tokenizer = LegalitySyllableTokenizer(words.words())
        tokens = tokenizer.tokenize(test_word)
        assert tokens == ['won', 'der', 'ful']

    @check_stanford_segmenter
    def test_stanford_segmenter_arabic(self):
        """
        Test the Stanford Word Segmenter for Arabic (default config)
        """
        seg = StanfordSegmenter()
        seg.default_config('ar')
        sent = 'يبحث علم الحاسوب استخدام الحوسبة بجميع اشكالها لحل المشكلات'
        segmented_sent = seg.segment(sent.split())
        assert segmented_sent.split() == ['يبحث', 'علم', 'الحاسوب', 'استخدام', 'الحوسبة', 'ب', 'جميع', 'اشكال', 'ها', 'ل', 'حل', 'المشكلات']

    @check_stanford_segmenter
    def test_stanford_segmenter_chinese(self):
        """
        Test the Stanford Word Segmenter for Chinese (default config)
        """
        seg = StanfordSegmenter()
        seg.default_config('zh')
        sent = '这是斯坦福中文分词器测试'
        segmented_sent = seg.segment(sent.split())
        assert segmented_sent.split() == ['这', '是', '斯坦福', '中文', '分词器', '测试']

    def test_phone_tokenizer(self):
        """
        Test a string that resembles a phone number but contains a newline
        """
        tokenizer = TweetTokenizer()
        test1 = '(393)  928 -3010'
        expected = ['(393)  928 -3010']
        result = tokenizer.tokenize(test1)
        assert result == expected
        test2 = '(393)\n928 -3010'
        expected = ['(', '393', ')', '928 -3010']
        result = tokenizer.tokenize(test2)
        assert result == expected

    def test_emoji_tokenizer(self):
        """
        Test a string that contains Emoji ZWJ Sequences and skin tone modifier
        """
        tokenizer = TweetTokenizer()
        test1 = '👨\u200d👩\u200d👧\u200d👧'
        expected = ['👨\u200d👩\u200d👧\u200d👧']
        result = tokenizer.tokenize(test1)
        assert result == expected
        test2 = '👨🏿'
        expected = ['👨🏿']
        result = tokenizer.tokenize(test2)
        assert result == expected
        test3 = '🤔 🙈 me así, se😌 ds 💕👭👙 hello 👩🏾\u200d🎓 emoji hello 👨\u200d👩\u200d👦\u200d👦 how are 😊 you today🙅🏽🙅🏽'
        expected = ['🤔', '🙈', 'me', 'así', ',', 'se', '😌', 'ds', '💕', '👭', '👙', 'hello', '👩🏾\u200d🎓', 'emoji', 'hello', '👨\u200d👩\u200d👦\u200d👦', 'how', 'are', '😊', 'you', 'today', '🙅🏽', '🙅🏽']
        result = tokenizer.tokenize(test3)
        assert result == expected
        test4 = '🇦🇵🇵🇱🇪'
        expected = ['🇦🇵', '🇵🇱', '🇪']
        result = tokenizer.tokenize(test4)
        assert result == expected
        test5 = 'Hi 🇨🇦, 😍!!'
        expected = ['Hi', '🇨🇦', ',', '😍', '!', '!']
        result = tokenizer.tokenize(test5)
        assert result == expected
        test6 = '<3 🇨🇦 🤝 🇵🇱 <3'
        expected = ['<3', '🇨🇦', '🤝', '🇵🇱', '<3']
        result = tokenizer.tokenize(test6)
        assert result == expected

    def test_pad_asterisk(self):
        """
        Test padding of asterisk for word tokenization.
        """
        text = 'This is a, *weird sentence with *asterisks in it.'
        expected = ['This', 'is', 'a', ',', '*', 'weird', 'sentence', 'with', '*', 'asterisks', 'in', 'it', '.']
        assert word_tokenize(text) == expected

    def test_pad_dotdot(self):
        """
        Test padding of dotdot* for word tokenization.
        """
        text = 'Why did dotdot.. not get tokenized but dotdotdot... did? How about manydots.....'
        expected = ['Why', 'did', 'dotdot', '..', 'not', 'get', 'tokenized', 'but', 'dotdotdot', '...', 'did', '?', 'How', 'about', 'manydots', '.....']
        assert word_tokenize(text) == expected

    def test_remove_handle(self):
        """
        Test remove_handle() from casual.py with specially crafted edge cases
        """
        tokenizer = TweetTokenizer(strip_handles=True)
        test1 = '@twitter hello @twi_tter_. hi @12345 @123news'
        expected = ['hello', '.', 'hi']
        result = tokenizer.tokenize(test1)
        assert result == expected
        test2 = '@n`@n~@n(@n)@n-@n=@n+@n\\@n|@n[@n]@n{@n}@n;@n:@n\'@n"@n/@n?@n.@n,@n<@n>@n @n\n@n ñ@n.ü@n.ç@n.'
        expected = ['`', '~', '(', ')', '-', '=', '+', '\\', '|', '[', ']', '{', '}', ';', ':', "'", '"', '/', '?', '.', ',', '<', '>', 'ñ', '.', 'ü', '.', 'ç', '.']
        result = tokenizer.tokenize(test2)
        assert result == expected
        test3 = 'a@n j@n z@n A@n L@n Z@n 1@n 4@n 7@n 9@n 0@n _@n !@n @@n #@n $@n %@n &@n *@n'
        expected = ['a', '@n', 'j', '@n', 'z', '@n', 'A', '@n', 'L', '@n', 'Z', '@n', '1', '@n', '4', '@n', '7', '@n', '9', '@n', '0', '@n', '_', '@n', '!', '@n', '@', '@n', '#', '@n', '$', '@n', '%', '@n', '&', '@n', '*', '@n']
        result = tokenizer.tokenize(test3)
        assert result == expected
        test4 = '@n!a @n#a @n$a @n%a @n&a @n*a'
        expected = ['!', 'a', '#', 'a', '$', 'a', '%', 'a', '&', 'a', '*', 'a']
        result = tokenizer.tokenize(test4)
        assert result == expected
        test5 = '@n!@n @n#@n @n$@n @n%@n @n&@n @n*@n @n@n @@n @n@@n @n_@n @n7@n @nj@n'
        expected = ['!', '@n', '#', '@n', '$', '@n', '%', '@n', '&', '@n', '*', '@n', '@n', '@n', '@', '@n', '@n', '@', '@n', '@n_', '@n', '@n7', '@n', '@nj', '@n']
        result = tokenizer.tokenize(test5)
        assert result == expected
        test6 = '@abcdefghijklmnopqrstuvwxyz @abcdefghijklmno1234 @abcdefghijklmno_ @abcdefghijklmnoendofhandle'
        expected = ['pqrstuvwxyz', '1234', '_', 'endofhandle']
        result = tokenizer.tokenize(test6)
        assert result == expected
        test7 = '@abcdefghijklmnop@abcde @abcdefghijklmno@abcde @abcdefghijklmno_@abcde @abcdefghijklmno5@abcde'
        expected = ['p', '@abcde', '@abcdefghijklmno', '@abcde', '_', '@abcde', '5', '@abcde']
        result = tokenizer.tokenize(test7)
        assert result == expected

    def test_treebank_span_tokenizer(self):
        """
        Test TreebankWordTokenizer.span_tokenize function
        """
        tokenizer = TreebankWordTokenizer()
        test1 = 'Good muffins cost $3.88\nin New (York).  Please (buy) me\ntwo of them.\n(Thanks).'
        expected = [(0, 4), (5, 12), (13, 17), (18, 19), (19, 23), (24, 26), (27, 30), (31, 32), (32, 36), (36, 37), (37, 38), (40, 46), (47, 48), (48, 51), (51, 52), (53, 55), (56, 59), (60, 62), (63, 68), (69, 70), (70, 76), (76, 77), (77, 78)]
        result = list(tokenizer.span_tokenize(test1))
        assert result == expected
        test2 = 'The DUP is similar to the "religious right" in the United States and takes a hardline stance on social issues'
        expected = [(0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42), (42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 85), (86, 92), (93, 95), (96, 102), (103, 109)]
        result = list(tokenizer.span_tokenize(test2))
        assert result == expected
        test3 = 'The DUP is similar to the "religious right" in the United States and takes a ``hardline\'\' stance on social issues'
        expected = [(0, 3), (4, 7), (8, 10), (11, 18), (19, 21), (22, 25), (26, 27), (27, 36), (37, 42), (42, 43), (44, 46), (47, 50), (51, 57), (58, 64), (65, 68), (69, 74), (75, 76), (77, 79), (79, 87), (87, 89), (90, 96), (97, 99), (100, 106), (107, 113)]
        result = list(tokenizer.span_tokenize(test3))
        assert result == expected

    def test_word_tokenize(self):
        """
        Test word_tokenize function
        """
        sentence = "The 'v', I've been fooled but I'll seek revenge."
        expected = ['The', "'", 'v', "'", ',', 'I', "'ve", 'been', 'fooled', 'but', 'I', "'ll", 'seek', 'revenge', '.']
        assert word_tokenize(sentence) == expected
        sentence = "'v' 're'"
        expected = ["'", 'v', "'", "'re", "'"]
        assert word_tokenize(sentence) == expected

    def test_punkt_pair_iter(self):
        test_cases = [('12', [('1', '2'), ('2', None)]), ('123', [('1', '2'), ('2', '3'), ('3', None)]), ('1234', [('1', '2'), ('2', '3'), ('3', '4'), ('4', None)])]
        for test_input, expected_output in test_cases:
            actual_output = [x for x in punkt._pair_iter(test_input)]
            assert actual_output == expected_output

    def test_punkt_pair_iter_handles_stop_iteration_exception(self):
        it = iter([])
        gen = punkt._pair_iter(it)
        list(gen)

    def test_punkt_tokenize_words_handles_stop_iteration_exception(self):
        obj = punkt.PunktBaseClass()

        class TestPunktTokenizeWordsMock:

            def word_tokenize(self, s):
                return iter([])
        obj._lang_vars = TestPunktTokenizeWordsMock()
        list(obj._tokenize_words('test'))

    def test_punkt_tokenize_custom_lang_vars(self):

        class BengaliLanguageVars(punkt.PunktLanguageVars):
            sent_end_chars = ('.', '?', '!', '।')
        obj = punkt.PunktSentenceTokenizer(lang_vars=BengaliLanguageVars())
        sentences = 'উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।'
        expected = ['উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন।', 'অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন।', 'এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।']
        assert obj.tokenize(sentences) == expected

    def test_punkt_tokenize_no_custom_lang_vars(self):
        obj = punkt.PunktSentenceTokenizer()
        sentences = 'উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।'
        expected = ['উপরাষ্ট্রপতি শ্রী এম ভেঙ্কাইয়া নাইডু সোমবার আই আই টি দিল্লির হীরক জয়ন্তী উদযাপনের উদ্বোধন করেছেন। অনলাইনের মাধ্যমে এই অনুষ্ঠানে কেন্দ্রীয় মানব সম্পদ উন্নয়নমন্ত্রী শ্রী রমেশ পোখরিয়াল ‘নিশাঙ্ক’  উপস্থিত ছিলেন। এই উপলক্ষ্যে উপরাষ্ট্রপতি হীরকজয়ন্তীর লোগো এবং ২০৩০-এর জন্য প্রতিষ্ঠানের লক্ষ্য ও পরিকল্পনার নথি প্রকাশ করেছেন।']
        assert obj.tokenize(sentences) == expected

    @pytest.mark.parametrize('input_text,n_sents,n_splits,lang_vars', [('Subject: Some subject. Attachments: Some attachments', 2, 1), ('Subject: Some subject! Attachments: Some attachments', 2, 1), ('This is just a normal sentence, just like any other.', 1, 0)])
    def punkt_debug_decisions(self, input_text, n_sents, n_splits, lang_vars=None):
        tokenizer = punkt.PunktSentenceTokenizer()
        if lang_vars != None:
            tokenizer._lang_vars = lang_vars
        assert len(tokenizer.tokenize(input_text)) == n_sents
        assert len(list(tokenizer.debug_decisions(input_text))) == n_splits

    def test_punkt_debug_decisions_custom_end(self):

        class ExtLangVars(punkt.PunktLanguageVars):
            sent_end_chars = ('.', '?', '!', '^')
        self.punkt_debug_decisions('Subject: Some subject^ Attachments: Some attachments', n_sents=2, n_splits=1, lang_vars=ExtLangVars())

    @pytest.mark.parametrize('sentences, expected', [('this is a test. . new sentence.', ['this is a test.', '.', 'new sentence.']), ('This. . . That', ['This.', '.', '.', 'That']), ('This..... That', ['This..... That']), ('This... That', ['This... That']), ('This.. . That', ['This.. .', 'That']), ('This. .. That', ['This.', '.. That']), ('This. ,. That', ['This.', ',.', 'That']), ('This!!! That', ['This!!!', 'That']), ('This! That', ['This!', 'That']), ("1. This is R .\n2. This is A .\n3. That's all", ['1.', 'This is R .', '2.', 'This is A .', '3.', "That's all"]), ("1. This is R .\t2. This is A .\t3. That's all", ['1.', 'This is R .', '2.', 'This is A .', '3.', "That's all"]), ('Hello.\tThere', ['Hello.', 'There'])])
    def test_sent_tokenize(self, sentences: str, expected: List[str]):
        assert sent_tokenize(sentences) == expected