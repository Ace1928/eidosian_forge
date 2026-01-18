import pytest
from kivy.tests.common import GraphicUnitTest
from kivy.base import EventLoop
from kivy.uix.bubble import Bubble
from kivy.uix.bubble import BubbleContent
from kivy.uix.bubble import BubbleButton
class BubbleTest(GraphicUnitTest):

    def move_frames(self, t):
        for i in range(t):
            EventLoop.idle()

    def test_no_content(self):
        bubble = Bubble()
        self.render(bubble)

    def test_add_remove_content(self):
        bubble = Bubble()
        content = BubbleContent()
        bubble.add_widget(content)
        self.render(bubble)
        bubble.remove_widget(content)
        self.render(bubble)

    def test_add_arbitrary_content(self):
        from kivy.uix.gridlayout import GridLayout
        bubble = Bubble()
        content = GridLayout()
        bubble.add_widget(content)
        self.render(bubble)

    def test_add_two_content_widgets_fails(self):
        from kivy.uix.bubble import BubbleException
        bubble = Bubble()
        content_1 = BubbleContent()
        content_2 = BubbleContent()
        bubble.add_widget(content_1)
        with self.assertRaises(BubbleException):
            bubble.add_widget(content_2)

    def test_add_content_with_buttons(self):
        bubble = Bubble()
        content = BubbleContent()
        content.add_widget(BubbleButton(text='Option A'))
        content.add_widget(BubbleButton(text='Option B'))
        bubble.add_widget(content)
        self.render(bubble)

    def assertSequenceAlmostEqual(self, seq1, seq2, delta=None):
        assert len(seq1) == len(seq2)
        for a, b in zip(seq1, seq2):
            self.assertAlmostEqual(a, b, delta=delta)

    def assertTestBubbleLayoutWithPredefinedArrowPos(self, bubble):
        arrow_length = bubble.arrow_length
        arrow_width = bubble.arrow_width
        bubble_width = bubble.test_bubble_width
        button_height = bubble.test_button_height
        expected_content_size = {'bottom_left': (bubble_width, button_height), 'bottom_mid': (bubble_width, button_height), 'bottom_right': (bubble_width, button_height), 'top_left': (bubble_width, button_height), 'top_mid': (bubble_width, button_height), 'top_right': (bubble_width, button_height), 'left_top': (bubble_width - arrow_length, button_height), 'left_mid': (bubble_width - arrow_length, button_height), 'left_bottom': (bubble_width - arrow_length, button_height), 'right_top': (bubble_width - arrow_length, button_height), 'right_mid': (bubble_width - arrow_length, button_height), 'right_bottom': (bubble_width - arrow_length, button_height)}[bubble.arrow_pos]
        self.assertSequenceAlmostEqual(bubble.content.size, expected_content_size)
        expected_arrow_layout_size = {'bottom_left': (bubble_width, arrow_length), 'bottom_mid': (bubble_width, arrow_length), 'bottom_right': (bubble_width, arrow_length), 'top_left': (bubble_width, arrow_length), 'top_mid': (bubble_width, arrow_length), 'top_right': (bubble_width, arrow_length), 'left_top': (arrow_length, button_height), 'left_mid': (arrow_length, button_height), 'left_bottom': (arrow_length, button_height), 'right_top': (arrow_length, button_height), 'right_mid': (arrow_length, button_height), 'right_bottom': (arrow_length, button_height)}[bubble.arrow_pos]
        self.assertSequenceAlmostEqual(bubble.arrow_layout_size, expected_arrow_layout_size)
        expected_content_position = {'bottom_left': (0, arrow_length), 'bottom_mid': (0, arrow_length), 'bottom_right': (0, arrow_length), 'top_left': (0, 0), 'top_mid': (0, 0), 'top_right': (0, 0), 'left_top': (arrow_length, 0), 'left_mid': (arrow_length, 0), 'left_bottom': (arrow_length, 0), 'right_top': (0, 0), 'right_mid': (0, 0), 'right_bottom': (0, 0)}[bubble.arrow_pos]
        self.assertSequenceAlmostEqual(bubble.content.pos, expected_content_position)
        expected_arrow_layout_position = {'bottom_left': (0, 0), 'bottom_mid': (0, 0), 'bottom_right': (0, 0), 'top_left': (0, button_height), 'top_mid': (0, button_height), 'top_right': (0, button_height), 'left_top': (0, 0), 'left_mid': (0, 0), 'left_bottom': (0, 0), 'right_top': (bubble_width - arrow_length, 0), 'right_mid': (bubble_width - arrow_length, 0), 'right_bottom': (bubble_width - arrow_length, 0)}[bubble.arrow_pos]
        self.assertSequenceAlmostEqual(bubble.arrow_layout_pos, expected_arrow_layout_position)
        hal = arrow_length / 2
        x_offset = 0.05 * bubble_width
        y_offset = 0.05 * button_height
        expected_arrow_center_pos_within_arrow_layout = {'bottom_left': (x_offset + arrow_width / 2, hal), 'bottom_mid': (bubble_width / 2, hal), 'bottom_right': (bubble_width - arrow_width / 2 - x_offset, hal), 'top_left': (x_offset + arrow_width / 2, hal), 'top_mid': (bubble_width / 2, hal), 'top_right': (bubble_width - arrow_width / 2 - x_offset, hal), 'left_top': (hal, button_height - arrow_width / 2 - y_offset), 'left_mid': (hal, button_height / 2), 'left_bottom': (hal, y_offset + arrow_width / 2), 'right_top': (hal, button_height - arrow_width / 2 - y_offset), 'right_mid': (hal, button_height / 2), 'right_bottom': (hal, y_offset + arrow_width / 2)}[bubble.arrow_pos]
        self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, expected_arrow_center_pos_within_arrow_layout)
        expected_arrow_rotation = {'bottom_left': 0, 'bottom_mid': 0, 'bottom_right': 0, 'top_left': 180, 'top_mid': 180, 'top_right': 180, 'left_top': 270, 'left_mid': 270, 'left_bottom': 270, 'right_top': 90, 'right_mid': 90, 'right_bottom': 90}[bubble.arrow_pos]
        self.assertAlmostEqual(bubble.arrow_rotation, expected_arrow_rotation)

    def test_bubble_layout_with_predefined_arrow_pos(self):
        for params in bubble_layout_with_predefined_arrow_pos_test_params:
            bubble_width, button_height, arrow_pos = params
            with self.subTest():
                print('(bubble_width={}, button_height={}, arrow_pos={})'.format(*params))
                bubble = _TestBubble(arrow_pos=arrow_pos)
                bubble.size_hint = (None, None)
                bubble.test_bubble_width = bubble_width
                bubble.test_button_height = button_height

                def update_bubble_size(instance, value):
                    w = bubble_width
                    h = bubble.content_height + bubble.arrow_margin_y
                    bubble.size = (w, h)
                bubble.bind(content_size=update_bubble_size, arrow_margin=update_bubble_size)
                content = _TestBubbleContent()
                for i in range(3):
                    content.add_widget(_TestBubbleButton(button_size=(None, button_height), text='Option {}'.format(i)))
                bubble.add_widget(content)
                self.render(bubble)
                self.assertTestBubbleLayoutWithPredefinedArrowPos(bubble)

    def test_bubble_layout_without_arrow(self):
        bubble_width = 200
        button_height = 30
        bubble = _TestBubble(show_arrow=False)
        bubble.size_hint = (None, None)

        def update_bubble_size(instance, value):
            w = bubble_width
            h = bubble.content_height
            bubble.size = (w, h)
        bubble.bind(content_size=update_bubble_size)
        content = _TestBubbleContent(orientation='vertical')
        for i in range(7):
            content.add_widget(_TestBubbleButton(button_size=(None, button_height), text='Option_{}'.format(i)))
        bubble.add_widget(content)
        self.render(bubble)
        self.assertSequenceAlmostEqual(bubble.content.size, (bubble_width, 7 * button_height))
        self.assertSequenceAlmostEqual(bubble.content.pos, (0, 0))

    def test_bubble_layout_with_flex_arrow_pos(self):
        for params in bubble_layout_with_flex_arrow_pos_test_params:
            bubble_size = params[:2]
            flex_arrow_pos = params[2:4]
            arrow_side = params[4]
            with self.subTest():
                print('(w={}, h={}, x={}, y={}, side={})'.format(*params))
                bubble = _TestBubble()
                bubble.size_hint = (None, None)
                bubble.size = bubble_size
                bubble.flex_arrow_pos = flex_arrow_pos
                content = _TestBubbleContent(orientation='vertical')
                content.size_hint = (1, 1)
                button = _TestBubbleButton(button_size=(None, None), text='Option')
                button.size_hint_y = 1
                content.add_widget(button)
                bubble.add_widget(content)
                self.render(bubble)
                haw = bubble.arrow_width / 2
                if arrow_side in ['l', 'r']:
                    self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, (haw, flex_arrow_pos[1]), delta=haw)
                elif arrow_side in ['b', 't']:
                    self.assertSequenceAlmostEqual(bubble.arrow_center_pos_within_arrow_layout, (flex_arrow_pos[0], haw), delta=haw)