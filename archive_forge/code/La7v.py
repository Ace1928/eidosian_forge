import speech_recognition as sr

# Create a recognizer object
recognizer = sr.Recognizer()

# Load the audio file
with sr.AudioFile("audio.wav") as source:
    # Read the audio data
    audio = recognizer.record(source)

# Perform speech recognition
try:
    text = recognizer.recognize_google(audio)
    print("Transcription:", text)
except sr.UnknownValueError:
    print("Speech recognition could not understand the audio.")
except sr.RequestError as e:
    print(
        "Could not request results from the speech recognition service; {0}".format(e)
    )
