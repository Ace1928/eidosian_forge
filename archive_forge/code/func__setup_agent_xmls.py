import traceback
from copy import deepcopy
import json
import logging
from minerl.env.comms import retry
from minerl.env.exceptions import MissionInitException
import os
from minerl.herobraine.wrapper import EnvWrapper
import struct
from minerl.env.malmo import InstanceManager, MinecraftInstance, launch_queue_logger_thread, malmo_version
import uuid
import coloredlogs
import gym
import socket
import time
from lxml import etree
from minerl.env import comms
import xmltodict
from concurrent.futures import ThreadPoolExecutor
import cv2
from minerl.herobraine.env_spec import EnvSpec
from typing import Any, Callable, Dict, List, Optional, Tuple
def _setup_agent_xmls(self, ep_uid: str) -> List[etree.Element]:
    """Generates the XML for an episode.

        THIS SHOULD EVENTUALLY BE DEPRECATED FOR FULL JINJA TEMPALTING!

        Returns:
            str: The XML for an episode.
        """
    xml_in = self.task.to_xml()
    agent_xmls = []
    base_xml = etree.fromstring(xml_in)
    for role in range(self.task.agent_count):
        agent_xml = deepcopy(base_xml)
        agent_xml_etree = etree.fromstring('<MissionInit xmlns="http://ProjectMalmo.microsoft.com"\n                   xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"\n                   SchemaVersion="" PlatformVersion=' + '"' + malmo_version + '"' + '>\n                <ExperimentUID>{ep_uid}</ExperimentUID>\n                <ClientRole>{role}</ClientRole>\n                <ClientAgentConnection>\n                <ClientIPAddress>127.0.0.1</ClientIPAddress>\n                <ClientMissionControlPort>0</ClientMissionControlPort>\n                <ClientCommandsPort>0</ClientCommandsPort>\n                <AgentIPAddress>127.0.0.1</AgentIPAddress>\n                <AgentMissionControlPort>0</AgentMissionControlPort>\n                <AgentVideoPort>0</AgentVideoPort>\n                <AgentDepthPort>0</AgentDepthPort>\n                <AgentLuminancePort>0</AgentLuminancePort>\n                <AgentObservationsPort>0</AgentObservationsPort>\n                <AgentRewardsPort>0</AgentRewardsPort>\n                <AgentColourMapPort>0</AgentColourMapPort>\n                </ClientAgentConnection>\n             </MissionInit>'.format(ep_uid=ep_uid, role=role))
        agent_xml_etree.insert(0, agent_xml)
        if self._is_interacting and role == 0:
            hi = etree.fromstring('\n                    <HumanInteraction>\n                        <Port>{}</Port>\n                        <MaxPlayers>{}</MaxPlayers>\n                    </HumanInteraction>'.format(self.interact_port, self.max_players))
            namespace = '{http://ProjectMalmo.microsoft.com}'
            ss = agent_xml_etree.find('.//' + namespace + 'ServerSection')
            ss.insert(0, hi)
        xml_dict = self._xml_mutator_to_be_deprecated(xmltodict.parse(etree.tostring(agent_xml_etree)))
        agent_xml_etree = etree.fromstring(xmltodict.unparse(xml_dict).encode())
        agent_xmls.append(agent_xml_etree)
    return agent_xmls