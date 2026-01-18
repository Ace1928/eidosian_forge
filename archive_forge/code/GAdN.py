from gtts import gTTS
import os

# Text to convert to speech
text = "Hello, this is an example of text-to-speech conversion using Python."

# Create a gTTS object
tts = gTTS(text=text, lang="en")

# Define the path for the audio file
audio_path = "/home/lloyd/Downloads/audio"
audio_file = os.path.join(audio_path, "output.mp3")

# Ensure the directory exists
os.makedirs(audio_path, exist_ok=True)

# Save the speech as an audio file
tts.save(audio_file)

# Play the audio file
os.system(f"start {audio_file}")
