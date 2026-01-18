from __future__ import absolute_import
import os
import sys
import socket
import struct
import subprocess
import argparse
import time
import logging
from threading import Thread
def accept_slaves(self, nslave):
    shutdown = {}
    wait_conn = {}
    job_map = {}
    pending = []
    tree_map = None
    while len(shutdown) != nslave:
        fd, s_addr = self.sock.accept()
        s = SlaveEntry(fd, s_addr)
        if s.cmd == 'print':
            msg = s.sock.recvstr()
            logging.info(msg.strip())
            continue
        if s.cmd == 'shutdown':
            assert s.rank >= 0 and s.rank not in shutdown
            assert s.rank not in wait_conn
            shutdown[s.rank] = s
            logging.debug('Recieve %s signal from %d', s.cmd, s.rank)
            continue
        assert s.cmd == 'start' or s.cmd == 'recover'
        if tree_map is None:
            assert s.cmd == 'start'
            if s.world_size > 0:
                nslave = s.world_size
            tree_map, parent_map, ring_map = self.get_link_map(nslave)
            todo_nodes = list(range(nslave))
        else:
            assert s.world_size == -1 or s.world_size == nslave
        if s.cmd == 'recover':
            assert s.rank >= 0
        rank = s.decide_rank(job_map)
        if rank == -1:
            assert len(todo_nodes) != 0
            pending.append(s)
            if len(pending) == len(todo_nodes):
                pending.sort(key=lambda x: x.host)
                for s in pending:
                    rank = todo_nodes.pop(0)
                    if s.jobid != 'NULL':
                        job_map[s.jobid] = rank
                    s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                    if s.wait_accept > 0:
                        wait_conn[rank] = s
                    logging.debug('Recieve %s signal from %s; assign rank %d', s.cmd, s.host, s.rank)
            if len(todo_nodes) == 0:
                logging.info('@tracker All of %d nodes getting started', nslave)
                self.start_time = time.time()
        else:
            s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
            logging.debug('Recieve %s signal from %d', s.cmd, s.rank)
            if s.wait_accept > 0:
                wait_conn[rank] = s
    logging.info('@tracker All nodes finishes job')
    self.end_time = time.time()
    logging.info('@tracker %s secs between node start and job finish', str(self.end_time - self.start_time))