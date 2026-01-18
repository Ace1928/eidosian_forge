import json
import logging
import multiprocessing
import os
from collections import OrderedDict
from queue import Queue, PriorityQueue
from typing import List, Tuple, Any
import cv2
import numpy as np
from multiprocess.pool import Pool
from minerl.herobraine.hero.agent_handler import HandlerCollection, AgentHandler
from minerl.herobraine.hero.handlers import RewardHandler
@staticmethod
def _get_load_data_func(data_queue, nsteps, worker_batch_size, mission_handlers, observables, actionables, gamma):

    def _load_data(inst_dir):
        recording_path = str(os.path.join(inst_dir, 'recording.mp4'))
        univ_path = str(os.path.join(inst_dir, 'univ.json'))
        try:
            cap = cv2.VideoCapture(recording_path)
            with open(univ_path, 'r') as f:
                univ = {int(k): v for k, v in json.load(f).items()}
                univ = OrderedDict(univ)
                univ = np.array(list(univ.values()))
            batches = []
            rewards = []
            frames_queue = Queue(maxsize=nsteps)
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret or frame_num >= len(univ):
                    break
                else:
                    if len(batches) >= worker_batch_size:
                        data_queue.put(batches)
                        batches = []
                    try:
                        vf = np.clip(frame[:, :, ::-1], 0, 255)
                        uf = univ[frame_num]
                        frame = {'pov': vf}
                        frame.update(uf)
                        cur_reward = 0
                        for m in mission_handlers:
                            try:
                                if isinstance(m, RewardHandler):
                                    cur_reward += m.from_universal(frame)
                            except NotImplementedError:
                                pass
                        rewards.append(cur_reward)
                        frames_queue.put(frame)
                        if frames_queue.full():
                            next_obs = [o.from_universal(frame) for o in observables]
                            frame = frames_queue.get()
                            obs = [o.from_universal(frame) for o in observables]
                            act = [a.from_universal(frame) for a in actionables]
                            mission = []
                            for m in mission_handlers:
                                try:
                                    mission.append(m.from_universal(frame))
                                except NotImplementedError:
                                    mission.append(None)
                                    pass
                            batches.append((obs, act, mission, next_obs, DataPipelineWithReward._calculate_discount_rew(rewards[-nsteps:], gamma), frame_num + 1 - nsteps))
                    except Exception as e:
                        logger.warn('Exception {} caught in the middle of parsing {} in a worker of the data pipeline.'.format(e, inst_dir))
                frame_num += 1
            return batches
        except Exception as e:
            logger.error('Caught Exception')
            raise e
            return None
    return _load_data