from argparse import ArgumentParser
import json
from parlai.projects.self_feeding.utils import extract_fb_episodes, episode_to_examples

    Converts a Fbdialog file of episodes into a json file of Parley examples.
    