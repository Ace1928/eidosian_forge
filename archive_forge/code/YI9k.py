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

# Setup detailed logging for operational insights and debugging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Pygame mixer with high-quality audio settings for optimal performance
pygame.init()
pygame.mixer.init(frequency=96000, size=-16, channels=2, buffer=512)
logging.info(
    "Pygame mixer initialized with high-quality audio settings for optimal performance."
)

# Constants and configurations
NOTE_FREQUENCIES: Dict[str, float] = {
    "C": 261.63,
    "C#": 277.18,
    "D": 293.66,
    "D#": 311.13,
    "E": 329.63,
    "F": 349.23,
    "F#": 369.99,
    "G": 392.00,
    "G#": 415.30,
    "A": 440.00,
    "A#": 466.16,
    "B": 493.88,
}
OCTAVE_RANGE: range = range(0, 8)
sound_cache: Dict[str, pygame.mixer.Sound] = {}


# Function to generate high-quality sound array using specified timbre and harmonics
def generate_sound_array(
    frequency: float,
    duration: float,
    volume: float,
    harmonics_weights: List[float],
    timbre: str,
) -> np.ndarray:
    fs: int = 96000  # Sampling rate
    t: np.ndarray = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone: np.ndarray = np.zeros_like(t)
    for i, weight in enumerate(harmonics_weights):
        harmonic_frequency: float = (i + 1) * frequency
        if timbre == "sine":
            tone += weight * np.sin(2 * np.pi * harmonic_frequency * t)
        elif timbre == "square":
            tone += weight * np.sign(np.sin(2 * np.pi * harmonic_frequency * t))
        elif timbre == "sawtooth":
            tone += weight * 2 * (t * harmonic_frequency % 1 - 0.5)
    sound_array: np.ndarray = np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    sound_array = sound_array.astype(np.int16)
    return sound_array


# Function to save sound array to a WAV file
def save_sound_to_file(sound_array: np.ndarray, filename: str) -> None:
    with wave.open(filename, "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(96000)
        wf.writeframes(sound_array.tobytes())
        logging.info(f"Saved {filename} to disk with high-quality audio settings.")


# Function to generate and save notes across octaves
def generate_and_save_notes() -> None:
    if not os.path.exists("sounds"):
        os.makedirs("sounds")
    for octave in OCTAVE_RANGE:
        for note, freq in NOTE_FREQUENCIES.items():
            frequency: float = freq * (
                2 ** (octave - 4)
            )  # Middle C (C4) as the reference octave
            filename: str = f"sounds/{note}{octave}.wav"
            if not os.path.exists(filename):
                sound_array: np.ndarray = generate_sound_array(
                    frequency, 1.0, 0.75, [1, 0.5, 0.25], "sine"
                )
                save_sound_to_file(sound_array, filename)
                sound_cache[filename] = pygame.mixer.Sound(filename)
                logging.info(f"Generated and cached {filename}")


# Function to load or define key frequencies
def load_or_define_frequencies() -> Dict[int, float]:
    path: str = "key_frequencies.json"
    try:
        with open(path, "r") as file:
            frequencies: Dict[int, float] = json.load(file)
            logging.info("Successfully loaded key frequencies from file.")
    except Exception as e:
        logging.error(
            f"Failed to load or generate key frequencies due to {e}. Generating default frequencies."
        )
        frequencies: Dict[int, float] = {
            pygame.K_a + i: 261.63 * 2 ** ((i - 9) / 12) for i in range(26)
        }  # A4 tuning
        with open(path, "w") as file:
            json.dump(frequencies, file, indent=4)
            logging.info("Default key frequencies written to file.")
    return frequencies


def frequency_to_note_name_and_octave(frequency: float) -> Tuple[str, int]:
    """
    Convert a frequency to a note name and octave number based on standard note frequencies.

    Args:
        frequency (float): The frequency to convert.

    Returns:
        Tuple[str, int]: A tuple containing the note name and octave number.

    Raises:
        ValueError: If the frequency is outside the range of defined note frequencies.
    """
    # Define the base frequency for the note A4
    A4 = 440.0
    # Calculate the number of half steps away from A4
    half_steps_from_A4 = 12 * np.log2(frequency / A4)
    # Round to the nearest whole number to find the closest note
    rounded_half_steps = int(round(half_steps_from_A4))
    # Calculate the actual note index by wrapping around the NOTE_FREQUENCIES dictionary
    note_index = (rounded_half_steps % 12 + 12) % 12
    # Determine the octave by dividing the total half steps by 12 and adjusting based on A4's octave
    octave = 4 + (rounded_half_steps // 12)
    # Find the note name by using the note index
    note_name = list(NOTE_FREQUENCIES.keys())[note_index]

    # Ensure the calculated octave is within the defined range
    if octave not in range(OCTAVE_RANGE.start, OCTAVE_RANGE.stop):
        raise ValueError(
            f"The frequency {frequency} Hz results in an octave {octave}, which is out of the defined range."
        )

    return note_name, octave


generate_and_save_notes()
key_to_frequency: Dict[int, float] = load_or_define_frequencies()

# GUI Setup
manager: pygame_gui.UIManager = pygame_gui.UIManager((800, 600))
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
