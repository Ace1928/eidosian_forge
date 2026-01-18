import pygame
import numpy as np
import json
from typing import Dict, Tuple, Any
import os
import sys

# Initialize Pygame mixer with optimal settings for performance and quality
pygame.init()
pygame.mixer.init(
    frequency=44100, size=-16, channels=2, buffer=512
)  # Stereo channel with reduced buffer size for lower latency


# Function to generate and play a tone with detailed control over its properties
def play_tone(frequency: float, duration: float, volume: float = 0.5) -> None:
    """
    Generate and play a tone using sine wave generation with specified frequency, duration, and volume.

    Args:
        frequency (float): Frequency of the tone in Hz.
        duration (float): Duration of the tone in seconds.
        volume (float): Volume of the tone, default is 0.5.

    Detailed implementation ensures high fidelity audio production.
    """
    fs = 44100  # Optimal sampling rate for audio quality
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    tone = volume * np.sin(2 * np.pi * frequency * t)
    sound = (
        np.asarray([tone] * 2).T * 32767 / np.max(np.abs(tone))
    )  # Normalize and convert to stereo sound
    sound = sound.astype(np.int16)  # Convert to 16-bit PCM
    sound_obj = pygame.sndarray.make_sound(sound)
    sound_obj.play(loops=0)
    pygame.time.delay(int(duration * 1000))
    sound_obj.stop()


# Function to load or define a mapping of keyboard keys to frequencies, with error handling and default settings
def load_or_define_frequencies() -> Dict[int, float]:
    """
    Load the key to frequency mapping from a JSON file or define default values if file not found.

    Returns:
        Dict[int, float]: Mapping of Pygame key constants to frequencies in Hz.
    """
    path = "key_frequencies.json"
    if os.path.exists(path):
        with open(path, "r") as file:
            key_to_frequency = json.load(file)
    else:
        key_to_frequency = {
            pygame.K_a: 261.63,  # C4
            pygame.K_s: 293.66,  # D4
            pygame.K_d: 329.63,  # E4
            pygame.K_f: 349.23,  # F4
            pygame.K_g: 392.00,  # G4
            pygame.K_h: 440.00,  # A4
            pygame.K_j: 493.88,  # B4
            pygame.K_k: 523.25,  # C5
        }
        with open(path, "w") as file:
            json.dump(key_to_frequency, file, indent=4)
    return key_to_frequency


key_to_frequency = load_or_define_frequencies()

# Enhanced graphical interface setup
screen = pygame.display.set_mode(
    (800, 600)
)  # Larger display for better user interaction
pygame.display.set_caption("Advanced Simple Synthesizer")


# Function to handle configuration changes with detailed user feedback
def configure_keys() -> None:
    """
    Allow the user to configure the frequency associated with each key interactively.
    """
    print(
        "Interactive Key Configuration Mode: Press the desired key to configure, or 'ESC' to exit."
    )
    configuring = True
    while configuring:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    configuring = False
                else:
                    try:
                        new_freq = float(
                            input(
                                f"Enter new frequency for {pygame.key.name(event.key)} (in Hz): "
                            )
                        )
                        key_to_frequency[event.key] = new_freq
                        with open("key_frequencies.json", "w") as file:
                            json.dump(key_to_frequency, file, indent=4)
                        print(
                            f"Frequency for {pygame.key.name(event.key)} set to {new_freq} Hz."
                        )
                    except ValueError:
                        print("Invalid input. Please enter a valid numeric frequency.")


# Main event loop with enhanced event handling for efficiency and user experience
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                configure_keys()
            elif event.key in key_to_frequency:
                play_tone(key_to_frequency[event.key], duration=0.5)

pygame.quit()
