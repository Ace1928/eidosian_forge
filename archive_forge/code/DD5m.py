import pygame
import numpy as np
import json
import os
import logging
from typing import Dict, Tuple, Any
import wave

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with optimal settings for performance and quality
pygame.init()
pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
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
OCTAVE_RANGE = range(0, 9)  # Octaves 0 to 8


# Generate and save sound files for all notes
def generate_and_save_notes() -> None:
    """
    Generate and save sound files for all notes across all octaves specified in OCTAVE_RANGE.
    Each note is saved as a WAV file in the 'sounds' directory.
    """
    for octave in OCTAVE_RANGE:
        for note, base_freq in NOTE_FREQUENCIES.items():
            frequency = base_freq * (2**octave)
            filename = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array = generate_sound_array(frequency, 1.0)  # 1 second duration
                save_sound_to_file(sound_array, filename)
                logging.info(f"Generated and saved {filename}")


def generate_sound_array(
    frequency: float, duration: float, volume: float = 0.5
) -> np.ndarray:
    """
    Generate a sound array using sine wave generation with specified frequency, duration, and volume.
    """
    fs = 44100  # Sampling rate
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound_array = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


def save_sound_to_file(sound_array: np.ndarray, filename: str) -> None:
    """
    Save a sound array to a WAV file.
    """
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # Sample width in bytes
        wf.setframerate(44100)  # Frame rate
        wf.writeframes(sound_array.tobytes())


# Load or define frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Load the key to frequency mapping from a JSON file or define default values if file not found.
    The mapping is for one octave starting from middle C (C4).

    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    path = "key_frequencies.json"
    if os.path.exists(path):
        with open(path, "r") as file:
            return json.load(file)
    else:
        # Default mapping for one octave starting from middle C (C4)
        default_frequencies = {
            pygame.K_a + i: NOTE_FREQUENCIES[note] * (2**4)
            for i, note in enumerate(NOTE_FREQUENCIES)
        }
        with open(path, "w") as file:
            json.dump(default_frequencies, file, indent=4)
        return default_frequencies


# Ensure sound files directory exists
if not os.path.exists("sounds"):
    os.makedirs("sounds")
generate_and_save_notes()

key_to_frequency = load_or_define_frequencies()

# Main event loop
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Advanced Simple Synthesizer")
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in key_to_frequency:
                filename = f"sounds/{key_to_frequency[event.key]}.wav"
                sound_obj = pygame.mixer.Sound(filename)
                sound_obj.play()

pygame.quit()
logging.info("Pygame terminated.")
