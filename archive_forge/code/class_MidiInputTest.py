import unittest
import pygame
class MidiInputTest(unittest.TestCase):
    __tags__ = ['interactive']

    def setUp(self):
        import pygame.midi
        pygame.midi.init()
        in_id = pygame.midi.get_default_input_id()
        if in_id != -1:
            self.midi_input = pygame.midi.Input(in_id)
        else:
            self.midi_input = None

    def tearDown(self):
        if self.midi_input:
            self.midi_input.close()
        pygame.midi.quit()

    def test_Input(self):
        i = pygame.midi.get_default_input_id()
        if self.midi_input:
            self.assertEqual(self.midi_input.device_id, i)
        i = pygame.midi.get_default_output_id()
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, i)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, 9009)
        self.assertRaises(pygame.midi.MidiException, pygame.midi.Input, -1)
        self.assertRaises(TypeError, pygame.midi.Input, '1234')
        self.assertRaises(OverflowError, pygame.midi.Input, pow(2, 99))

    def test_poll(self):
        if not self.midi_input:
            self.skipTest('No midi Input device')
        self.assertFalse(self.midi_input.poll())
        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.poll)
        self.midi_input = None

    def test_read(self):
        if not self.midi_input:
            self.skipTest('No midi Input device')
        read = self.midi_input.read(5)
        self.assertEqual(read, [])
        pygame.midi.quit()
        self.assertRaises(RuntimeError, self.midi_input.read, 52)
        self.midi_input = None

    def test_close(self):
        if not self.midi_input:
            self.skipTest('No midi Input device')
        self.assertIsNotNone(self.midi_input._input)
        self.midi_input.close()
        self.assertIsNone(self.midi_input._input)