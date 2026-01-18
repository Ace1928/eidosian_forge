import argparse
import logging
import socket
import struct
import sys
from threading import Thread
from typing import Dict, List, Optional, Set, Tuple, Union
class WorkerEntry:
    """Hanlder to each worker."""

    def __init__(self, sock: socket.socket, s_addr: Tuple[str, int]):
        worker = ExSocket(sock)
        self.sock = worker
        self.host = get_some_ip(s_addr[0])
        magic = worker.recvint()
        assert magic == MAGIC_NUM, f'invalid magic number={magic} from {self.host}'
        worker.sendint(MAGIC_NUM)
        self.rank = worker.recvint()
        self.world_size = worker.recvint()
        self.task_id = worker.recvstr()
        self.cmd = worker.recvstr()
        self.wait_accept = 0
        self.port: Optional[int] = None

    def print(self, use_logger: bool) -> None:
        """Execute the print command from worker."""
        msg = self.sock.recvstr()
        if use_logger:
            logging.info(msg.strip())
        else:
            print(msg.strip(), flush=True)

    def decide_rank(self, job_map: Dict[str, int]) -> int:
        """Get the rank of current entry."""
        if self.rank >= 0:
            return self.rank
        if self.task_id != 'NULL' and self.task_id in job_map:
            return job_map[self.task_id]
        return -1

    def assign_rank(self, rank: int, wait_conn: Dict[int, 'WorkerEntry'], tree_map: _TreeMap, parent_map: Dict[int, int], ring_map: _RingMap) -> List[int]:
        """Assign the rank for current entry."""
        self.rank = rank
        nnset = set(tree_map[rank])
        rprev, next_rank = ring_map[rank]
        self.sock.sendint(rank)
        self.sock.sendint(parent_map[rank])
        self.sock.sendint(len(tree_map))
        self.sock.sendint(len(nnset))
        for r in nnset:
            self.sock.sendint(r)
        if rprev not in (-1, rank):
            nnset.add(rprev)
            self.sock.sendint(rprev)
        else:
            self.sock.sendint(-1)
        if next_rank not in (-1, rank):
            nnset.add(next_rank)
            self.sock.sendint(next_rank)
        else:
            self.sock.sendint(-1)
        return self._get_remote(wait_conn, nnset)

    def _get_remote(self, wait_conn: Dict[int, 'WorkerEntry'], nnset: Set[int]) -> List[int]:
        while True:
            ngood = self.sock.recvint()
            goodset = set()
            for _ in range(ngood):
                goodset.add(self.sock.recvint())
            assert goodset.issubset(nnset)
            badset = nnset - goodset
            conset = []
            for r in badset:
                if r in wait_conn:
                    conset.append(r)
            self.sock.sendint(len(conset))
            self.sock.sendint(len(badset) - len(conset))
            for r in conset:
                self.sock.sendstr(wait_conn[r].host)
                port = wait_conn[r].port
                assert port is not None
                self.sock.sendint(port)
                self.sock.sendint(r)
            nerr = self.sock.recvint()
            if nerr != 0:
                continue
            self.port = self.sock.recvint()
            rmset = []
            for r in conset:
                wait_conn[r].wait_accept -= 1
                if wait_conn[r].wait_accept == 0:
                    rmset.append(r)
            for r in rmset:
                wait_conn.pop(r, None)
            self.wait_accept = len(badset) - len(conset)
            return rmset