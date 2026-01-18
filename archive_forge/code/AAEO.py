import pygame
import numpy as np
import json
import os
import logging
from collections import defaultdict
import threading
import pygame_gui

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with enhanced settings for performance and quality
pygame.init()
pygame.mixer.init(frequency=96000, size=-16, channels=2, buffer=512)
logging.info("Pygame mixer initialized with high-quality audio settings.")

# Constants
NOTE_FREQUENCIES = {
    "C": 16.35,
    "C#": 17.32,
    "D": 18.35,
    "D#": 19.45,
    "E": 20.60,
    "F": 21.83,
    "F#": 23.12,
    "G": 24.50,
    "G#": 25.96,
    "A": 27.50,
    "A#": 29.14,
    "B": 30.87,
}
OCTAVE_RANGE = range(0, 8)

# Sound cache dictionary
sound_cache = {}


def generate_sound_array(frequency, duration, volume):
    """
    Generate a high-quality sound array using sine wave generation.
    """
    fs = 96000  # Higher sampling rate for better audio quality
    t = np.linspace(0, duration, int(fs * duration), False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound_array = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


def save_sound_to_file(sound_array, filename):
    """
    Save a sound array to a WAV file.
    """
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(96000)
        wf.writeframes(sound_array.tobytes())


def generate_and_save_notes():
    """
    Generate and save sound files for all notes across all octaves.
    """
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
    for octave in OCTAVE_RANGE:
        for note, base_freq in NOTE_FREQUENCIES.items():
            frequency = base_freq * (2**octave)
            filename = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array = generate_sound_array(frequency, 1.0, 0.75)
                save_sound_to_file(sound_array, filename)
                logging.info(f"Generated and saved {filename}")
            sound_cache[filename] = pygame.mixer.Sound(filename)


def frequency_to_note_name_and_octave(freq):
    for note, base_freq in NOTE_FREQUENCIES.items():
        for octave in OCTAVE_RANGE:
            if base_freq * (2**octave) == freq:
                return note, octave
    return None, None


def load_or_define_frequencies():
    path = "key_frequencies.json"
    try:
        with open(path, "r") as file:
            frequencies = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        frequencies = {
            pygame.K_a + i: NOTE_FREQUENCIES[note] * (2**4)
            for i, note in enumerate(NOTE_FREQUENCIES)
        }
        with open(path, "w") as file:
            json.dump(frequencies, file, indent=4)
    return frequencies


generate_and_save_notes()
key_to_frequency = load_or_define_frequencies()

# Setup the GUI manager
manager = pygame_gui.UIManager((800, 600))

# Create a simple synthesizer interface
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Advanced Simple Synthesizer")
clock = pygame.time.Clock()
running = True

# Event loop
while running:
    time_delta = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            frequency = key_to_frequency.get(event.key, 440)
            filename = f"sounds/{frequency_to_note_name_and_octave(frequency)}.wav"
            if filename in sound_cache:
                sound_cache[filename].play(-1)
        elif event.type == pygame.KEYUP:
            frequency = key_to_frequency.get(event.key, 440)
            filename = f"sounds/{frequency_to_note_name_and_octave(frequency)}.wav"
            if filename in sound_cache:
                sound_cache[filename].stop()

    manager.update(time_delta)
    screen.fill((0, 0, 0))
    manager.draw_ui(screen)
    pygame.display.update()

pygame.quit()
logging.info("Pygame terminated.")
