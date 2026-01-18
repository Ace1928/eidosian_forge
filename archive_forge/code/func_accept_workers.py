import argparse
import logging
import socket
import struct
import sys
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Union
def accept_workers(self, n_workers: int) -> None:
    """Wait for all workers to connect to the tracker."""
    shutdown: Dict[int, WorkerEntry] = {}
    wait_conn: Dict[int, WorkerEntry] = {}
    job_map: Dict[str, int] = {}
    pending: List[WorkerEntry] = []
    tree_map = None
    while len(shutdown) != n_workers:
        fd, s_addr = self.sock.accept()
        s = WorkerEntry(fd, s_addr)
        if s.cmd == 'print':
            s.print(self._use_logger)
            continue
        if s.cmd == 'shutdown':
            assert s.rank >= 0 and s.rank not in shutdown
            assert s.rank not in wait_conn
            shutdown[s.rank] = s
            logging.debug('Received %s signal from %d', s.cmd, s.rank)
            continue
        assert s.cmd in ('start', 'recover')
        if tree_map is None:
            assert s.cmd == 'start'
            if s.world_size > 0:
                n_workers = s.world_size
            tree_map, parent_map, ring_map = self.get_link_map(n_workers)
            todo_nodes = list(range(n_workers))
        else:
            assert s.world_size in (-1, n_workers)
        if s.cmd == 'recover':
            assert s.rank >= 0
        rank = s.decide_rank(job_map)
        if rank == -1:
            assert todo_nodes
            pending.append(s)
            if len(pending) == len(todo_nodes):
                pending = self._sort_pending(pending)
                for s in pending:
                    rank = todo_nodes.pop(0)
                    if s.task_id != 'NULL':
                        job_map[s.task_id] = rank
                    s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
                    if s.wait_accept > 0:
                        wait_conn[rank] = s
                    logging.debug('Received %s signal from %s; assign rank %d', s.cmd, s.host, s.rank)
            if not todo_nodes:
                logging.info('@tracker All of %d nodes getting started', n_workers)
        else:
            s.assign_rank(rank, wait_conn, tree_map, parent_map, ring_map)
            logging.debug('Received %s signal from %d', s.cmd, s.rank)
            if s.wait_accept > 0:
                wait_conn[rank] = s
    logging.info('@tracker All nodes finishes job')