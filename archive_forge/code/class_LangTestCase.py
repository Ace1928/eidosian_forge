import unittest
import os
from weakref import proxy
from functools import partial
from textwrap import dedent
class LangTestCase(unittest.TestCase):

    def import_builder(self):
        from kivy.factory import Factory
        from kivy.lang import BuilderBase
        Builder = BuilderBase()
        Factory.register('TLangClass', cls=TLangClass)
        Factory.register('TLangClass2', cls=TLangClass2)
        Factory.register('TLangClass3', cls=TLangClass3)
        return Builder

    def test_invalid_indentation(self):
        Builder = self.import_builder()
        from kivy.lang import ParserException
        kv_code = dedent("            BoxLayout:\n                orientation: 'vertical'\n                    Button:\n        ")
        try:
            Builder.load_string(kv_code)
            self.fail('Invalid indentation.')
        except ParserException:
            pass

    def test_invalid_indentation2(self):
        Builder = self.import_builder()
        from kivy.lang import ParserException
        kv_code = '   BoxLayout:'
        try:
            Builder.load_string(kv_code)
            self.fail('Invalid indentation.')
        except ParserException as e:
            pass

    def test_loading_failed_1(self):
        Builder = self.import_builder()
        from kivy.lang import ParserException
        try:
            Builder.load_string(dedent('#:kivy 1.0\n            <TLangClass>:\n            '))
            self.fail('Invalid indentation.')
        except ParserException:
            pass

    def test_parser_numeric_1(self):
        Builder = self.import_builder()
        Builder.load_string('<TLangClass>:\n\tobj: (.5, .5, .5)')
        wid = TLangClass()
        Builder.apply(wid)
        self.assertEqual(wid.obj, (0.5, 0.5, 0.5))

    def test_parser_numeric_2(self):
        Builder = self.import_builder()
        Builder.load_string('<TLangClass>:\n\tobj: (0.5, 0.5, 0.5)')
        wid = TLangClass()
        Builder.apply(wid)
        self.assertEqual(wid.obj, (0.5, 0.5, 0.5))

    def test_references(self):
        Builder = self.import_builder()
        Builder.load_string(dedent('\n        <TLangClass>:\n            textinput: textinput\n            TLangClass2:\n                id: textinput\n        '))
        wid = TLangClass()
        Builder.apply(wid)
        self.assertTrue(hasattr(wid, 'textinput'))
        self.assertTrue(getattr(wid, 'textinput') is not None)

    def test_references_with_template(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        [Item@TLangClass3]:\n            title: ctx.title\n        <TLangClass>:\n            textinput: textinput\n            Item:\n                title: 'bleh'\n            TLangClass2:\n                id: textinput\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        self.assertTrue(hasattr(wid, 'textinput'))
        self.assertTrue(getattr(wid, 'textinput') is not None)

    def test_references_with_template_case_2(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        [Item@TLangClass3]:\n            title: ctx.title\n        <TLangClass>:\n            textinput: textinput\n            TLangClass2:\n                id: textinput\n                Item:\n                    title: 'bleh'\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        self.assertTrue(hasattr(wid, 'textinput'))
        self.assertTrue(getattr(wid, 'textinput') is not None)

    def test_references_with_template_case_3(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        [Item@TLangClass3]:\n            title: ctx.title\n        <TLangClass>:\n            textinput: textinput\n            TLangClass2:\n                Item:\n                    title: 'bleh'\n                TLangClass2:\n                    TLangClass2:\n                        id: textinput\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        self.assertTrue(hasattr(wid, 'textinput'))
        self.assertTrue(getattr(wid, 'textinput') is not None)

    def test_with_multiline(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        <TLangClass>:\n            on_press:\n                print('hello world')\n                print('this is working !')\n                self.a = 1\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        wid.a = 0
        self.assertTrue('on_press' in wid.binded_func)
        wid.binded_func['on_press']()
        self.assertEqual(wid.a, 1)

    def test_with_eight_spaces(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        <TLangClass>:\n                on_press:\n                        print('hello world')\n                        print('this is working !')\n                        self.a = 1\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        wid.a = 0
        self.assertTrue('on_press' in wid.binded_func)
        wid.binded_func['on_press']()
        self.assertEqual(wid.a, 1)

    def test_with_one_space(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        <TLangClass>:\n         on_press:\n          print('hello world')\n          print('this is working !')\n          self.a = 1\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        wid.a = 0
        self.assertTrue('on_press' in wid.binded_func)
        wid.binded_func['on_press']()
        self.assertEqual(wid.a, 1)

    def test_with_two_spaces(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        <TLangClass>:\n          on_press:\n            print('hello world')\n            print('this is working !')\n            self.a = 1\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        wid.a = 0
        self.assertTrue('on_press' in wid.binded_func)
        wid.binded_func['on_press']()
        self.assertEqual(wid.a, 1)

    def test_property_trailingspace(self):
        Builder = self.import_builder()
        Builder.load_string(dedent("\n        <TLangClass>:\n            text : 'original'\n            on_press : self.text = 'changed'\n        "))
        wid = TLangClass()
        Builder.apply(wid)
        self.assertTrue('on_press' in wid.binded_func)
        self.assertEqual(wid.text, 'original')
        wid.binded_func['on_press']()
        self.assertEqual(wid.text, 'changed')

    def test_kv_python_init(self):
        from kivy.factory import Factory
        from kivy.lang import Builder
        from kivy.uix.widget import Widget

        class MyObject(object):
            value = 55

        class MyWidget(Widget):
            cheese = MyObject()
        Builder.load_string(dedent('\n        <MyWidget>:\n            x: 55\n            y: self.width + 10\n            height: self.cheese.value\n            width: 44\n\n        <MySecondWidget@Widget>:\n            x: 55\n            Widget:\n                x: 23\n        '))
        w = MyWidget(x=22, height=12, y=999)
        self.assertEqual(w.x, 22)
        self.assertEqual(w.width, 44)
        self.assertEqual(w.y, 44 + 10)
        self.assertEqual(w.height, 12)
        w2 = Factory.MySecondWidget(x=999)
        self.assertEqual(w2.x, 999)
        self.assertEqual(w2.children[0].x, 23)

    def test_apply_rules(self):
        Builder = self.import_builder()
        Builder.load_string('<TLangClassCustom>:\n\tobj: 42')
        wid = TLangClass()
        Builder.apply(wid)
        self.assertIsNone(wid.obj)
        Builder.apply_rules(wid, 'TLangClassCustom')
        self.assertEqual(wid.obj, 42)

    def test_load_utf8(self):
        from tempfile import mkstemp
        from kivy.lang import Builder
        fd, name = mkstemp()
        os.write(fd, dedent("\n        Label:\n            text: 'é 😊'\n        ").encode('utf8'))
        root = Builder.load_file(name)
        assert root.text == 'é 😊'
        os.close(fd)

    def test_bind_fstring(self):
        from kivy.lang import Builder
        label = Builder.load_string(dedent("\n        <TestLabel@Label>:\n            text: f'{self.pos}|{self.size}'\n        TestLabel:\n        "))
        assert label.text == '[0, 0]|[100, 100]'
        label.pos = (150, 200)
        assert label.text == '[150, 200]|[100, 100]'

    def test_bind_fstring_reference(self):
        from kivy.lang import Builder
        root = Builder.load_string(dedent("\n        FloatLayout:\n            Label:\n                id: original\n                text: 'perfect'\n            Label:\n                id: duplicate\n                text: f'{original.text}'\n        "))
        assert root.ids.duplicate.text == 'perfect'
        root.ids.original.text = 'new text'
        assert root.ids.duplicate.text == 'new text'

    def test_bind_fstring_expressions(self):
        from kivy.lang import Builder
        root = Builder.load_string(dedent('\n        FloatLayout:\n            Label:\n                id: original\n                text: \'perfect\'\n            Label:\n                id: target1\n                text: f"{\' \'.join(p.upper() for p in original.text)}"\n            Label:\n                id: target2\n                text: f"{\'\'.join(sorted({p.upper() for p in original.text}))}"\n            Label:\n                id: target3\n                text: f"{\'odd\' if len(original.text) % 2 else \'even\'}"\n            Label:\n                id: target4\n                text: f"{original.text[len(original.text) // 2:]}"\n            Label:\n                id: target5\n                text: f"{not len(original.text) % 2}"\n            Label:\n                id: target6\n                text: f"{original.text}" + " some text"\n        '))
        assert root.ids.target1.text == 'P E R F E C T'
        assert root.ids.target2.text == 'CEFPRT'
        assert root.ids.target3.text == 'odd'
        assert root.ids.target4.text == 'fect'
        assert root.ids.target5.text == 'False'
        assert root.ids.target6.text == 'perfect some text'
        root.ids.original.text = 'new text'
        assert root.ids.target1.text == 'N E W   T E X T'
        assert root.ids.target2.text == ' ENTWX'
        assert root.ids.target3.text == 'even'
        assert root.ids.target4.text == 'text'
        assert root.ids.target5.text == 'True'
        assert root.ids.target6.text == 'new text some text'

    def test_bind_fstring_expressions_should_not_bind(self):
        from kivy.lang import Builder
        root = Builder.load_string(dedent('\n        FloatLayout:\n            Label:\n                id: original\n                text: \'perfect\'\n            Label:\n                id: target1\n                text: f"{\' \'.join([original.text for _ in range(2)])}"\n            Label:\n                id: target2\n                text: f"{original.text.upper()}"\n            Label:\n                id: target3\n                text: f"{sum(obj.width for obj in root.children)}"\n        '))
        assert root.ids.target1.text == ' '
        assert root.ids.target2.text == ''
        assert root.ids.target3.text == '400'
        root.ids.original.text = 'new text'
        root.ids.original.width = 0
        assert root.ids.target1.text == ' '
        assert root.ids.target2.text == ''
        assert root.ids.target3.text == '400'