from midi2audio import FluidSynth
import argparse
import os
import subprocess
class FluidSynth:

    def __init__(self, sound_font=DEFAULT_SOUND_FONT, sample_rate=DEFAULT_SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.sound_font = os.path.expanduser(sound_font)

    def midi_to_audio(self, midi_file, audio_file):
        subprocess.call(['fluidsynth', '-ni', self.sound_font, midi_file, '-F', audio_file, '-r', str(self.sample_rate)])

    def play_midi(self, midi_file):
        subprocess.call(['fluidsynth', '-i', self.sound_font, midi_file, '-r', str(self.sample_rate)])