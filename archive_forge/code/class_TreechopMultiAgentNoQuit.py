from minerl.env.malmo import InstanceManager
from minerl.herobraine.env_specs.treechop_specs import Treechop
import gym
import minerl  # noqa
import argparse
import time
class TreechopMultiAgentNoQuit(Treechop):

    def create_server_quit_producers(self):
        return []