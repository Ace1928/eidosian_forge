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
class RabitTracker(object):
    """
    tracker for rabit
    """

    def __init__(self, hostIP, nslave, port=9091, port_end=9999):
        sock = socket.socket(get_family(hostIP), socket.SOCK_STREAM)
        for port in range(port, port_end):
            try:
                sock.bind((hostIP, port))
                self.port = port
                break
            except socket.error as e:
                if e.errno in [98, 48]:
                    continue
                else:
                    raise
        sock.listen(256)
        self.sock = sock
        self.hostIP = hostIP
        self.thread = None
        self.start_time = None
        self.end_time = None
        self.nslave = nslave
        logging.info('start listen on %s:%d', hostIP, self.port)

    def __del__(self):
        self.sock.close()

    @staticmethod
    def get_neighbor(rank, nslave):
        rank = rank + 1
        ret = []
        if rank > 1:
            ret.append(rank // 2 - 1)
        if rank * 2 - 1 < nslave:
            ret.append(rank * 2 - 1)
        if rank * 2 < nslave:
            ret.append(rank * 2)
        return ret

    def slave_envs(self):
        """
        get enviroment variables for slaves
        can be passed in as args or envs
        """
        return {'DMLC_TRACKER_URI': self.hostIP, 'DMLC_TRACKER_PORT': self.port}

    def get_tree(self, nslave):
        tree_map = {}
        parent_map = {}
        for r in range(nslave):
            tree_map[r] = self.get_neighbor(r, nslave)
            parent_map[r] = (r + 1) // 2 - 1
        return (tree_map, parent_map)

    def find_share_ring(self, tree_map, parent_map, r):
        """
        get a ring structure that tends to share nodes with the tree
        return a list starting from r
        """
        nset = set(tree_map[r])
        cset = nset - set([parent_map[r]])
        if len(cset) == 0:
            return [r]
        rlst = [r]
        cnt = 0
        for v in cset:
            vlst = self.find_share_ring(tree_map, parent_map, v)
            cnt += 1
            if cnt == len(cset):
                vlst.reverse()
            rlst += vlst
        return rlst

    def get_ring(self, tree_map, parent_map):
        """
        get a ring connection used to recover local data
        """
        assert parent_map[0] == -1
        rlst = self.find_share_ring(tree_map, parent_map, 0)
        assert len(rlst) == len(tree_map)
        ring_map = {}
        nslave = len(tree_map)
        for r in range(nslave):
            rprev = (r + nslave - 1) % nslave
            rnext = (r + 1) % nslave
            ring_map[rlst[r]] = (rlst[rprev], rlst[rnext])
        return ring_map

    def get_link_map(self, nslave):
        """
        get the link map, this is a bit hacky, call for better algorithm
        to place similar nodes together
        """
        tree_map, parent_map = self.get_tree(nslave)
        ring_map = self.get_ring(tree_map, parent_map)
        rmap = {0: 0}
        k = 0
        for i in range(nslave - 1):
            k = ring_map[k][1]
            rmap[k] = i + 1
        ring_map_ = {}
        tree_map_ = {}
        parent_map_ = {}
        for k, v in ring_map.items():
            ring_map_[rmap[k]] = (rmap[v[0]], rmap[v[1]])
        for k, v in tree_map.items():
            tree_map_[rmap[k]] = [rmap[x] for x in v]
        for k, v in parent_map.items():
            if k != 0:
                parent_map_[rmap[k]] = rmap[v]
            else:
                parent_map_[rmap[k]] = -1
        return (tree_map_, parent_map_, ring_map_)

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

    def start(self, nslave):

        def run():
            self.accept_slaves(nslave)
        self.thread = Thread(target=run, args=())
        self.thread.setDaemon(True)
        self.thread.start()

    def join(self):
        while self.thread.isAlive():
            self.thread.join(100)

    def alive(self):
        return self.thread.isAlive()