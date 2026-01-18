import pygame
import numpy as np
import json
import os
import logging
import threading
import pygame_gui
from typing import Dict, Tuple, Any, List
import wave
from collections import defaultdict

# Setup logging for detailed debugging and operational information
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with enhanced settings for performance and quality
pygame.init()
pygame.mixer.init(
    frequency=96000, size=-16, channels=2, buffer=512
)  # Enhanced audio settings
logging.info(
    "Pygame mixer initialized with high-quality audio settings and reduced buffer size for lower latency."
)

# Constants
NOTE_FREQUENCIES: Dict[str, float] = {
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
OCTAVE_RANGE: range = range(0, 8)  # Octaves 0 to 8
sound_cache: Dict[str, pygame.mixer.Sound] = {}


# Function to generate high-quality sound array using sine wave generation
def generate_sound_array(
    frequency: float, duration: float, volume: float = 0.75
) -> np.ndarray:
    """
    Generate a high-quality sound array using sine wave generation with specified frequency, duration, and volume.
    """
    fs: int = 96000  # Higher sampling rate for better audio quality
    t: np.ndarray = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone: np.ndarray = volume * np.sin(2 * np.pi * frequency * t)
    sound_array: np.ndarray = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


# Function to save sound array to a WAV file
def save_sound_to_file(sound_array: np.ndarray, filename: str) -> None:
    """
    Save a sound array to a WAV file.
    """
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(96000)
        wf.writeframes(sound_array.tobytes())


# Function to generate and save notes
def generate_and_save_notes() -> None:
    """
    Generate and save sound files for all notes across all octaves specified in OCTAVE_RANGE.
    Each note is saved as a WAV file in the 'sounds' directory.
    """
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
    for octave in OCTAVE_RANGE:
        for note, base_freq in NOTE_FREQUENCIES.items():
            frequency: float = base_freq * (2**octave)
            filename: str = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array: np.ndarray = generate_sound_array(frequency, 1.0, 0.75)
                save_sound_to_file(sound_array, filename)
                logging.info(f"Generated and saved {filename}")
            sound_cache[filename] = pygame.mixer.Sound(filename)


# Function to map frequency to note name and octave
def frequency_to_note_name_and_octave(freq: float) -> Tuple[str, int]:
    for note, base_freq in NOTE_FREQUENCIES.items():
        for octave in OCTAVE_RANGE:
            if base_freq * (2**octave) == freq:
                return note, octave
    return "", 0  # if no matching note is found


# Function to load or define frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Attempt to load the key to frequency mapping from a JSON file. If the file is not found or is corrupt,
    generate default values for two octaves starting from middle C (C4) and save them to the file, covering all keys from 'a' to 'z'.

    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    path: str = "key_frequencies.json"
    try:
        with open(path, "r") as file:
            frequencies: Dict[int, float] = json.load(file)
            logging.info("Successfully loaded key frequencies from file.")
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(
            f"Failed to load key frequencies due to {e}. Generating default frequencies."
        )
        # Define a sequence of notes for two octaves
        notes_sequence = (
            "C C# D D# E F F# G G# A A# B C C# D D# E F F# G G# A A# B".split()
        )
        # Generate frequencies for keys 'a' to 'z'
        frequencies: Dict[int, float] = {
            pygame.K_a + i: NOTE_FREQUENCIES[note] * (2**4)
            for i, note in enumerate(NOTE_FREQUENCIES)
        }
        with open(path, "w") as file:
            json.dump(frequencies, file, indent=4)
            logging.info("Default key frequencies written to file.")
    return frequencies


# Ensure sound files directory exists
if not os.path.exists("sounds"):
    os.makedirs("sounds")
generate_and_save_notes()

key_to_frequency: Dict[int, float] = load_or_define_frequencies()

# Setup the GUI manager
manager: pygame_gui.UIManager = pygame_gui.UIManager((800, 600))

# Create a simple synthesizer interface
screen: pygame.Surface = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Advanced Simple Synthesizer")
clock: pygame.time.Clock = pygame.time.Clock()
running: bool = True

# Main event loop
while running:
    time_delta: float = clock.tick(60) / 1000.0
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            frequency: float = key_to_frequency.get(event.key, 440)
            note_name, octave = frequency_to_note_name_and_octave(frequency)
            filename: str = f"sounds/{note_name}{octave}.wav"
            if filename in sound_cache:
                sound_cache[filename].play(-1)
        elif event.type == pygame.KEYUP:
            frequency: float = key_to_frequency.get(event.key, 440)
            note_name, octave = frequency_to_note_name_and_octave(frequency)
            filename: str = f"sounds/{note_name}{octave}.wav"
            if filename in sound_cache:
                sound_cache[filename].stop()

    manager.update(time_delta)
    screen.fill((0, 0, 0))
    manager.draw_ui(screen)
    pygame.display.update()

pygame.quit()
logging.info("Pygame terminated.")
