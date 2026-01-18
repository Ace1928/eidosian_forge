import asyncio
import logging
import ray.dashboard.consts as dashboard_consts
from ray.dashboard.utils import (
class DataOrganizer:
    head_node_ip = None

    @staticmethod
    @async_loop_forever(dashboard_consts.PURGE_DATA_INTERVAL_SECONDS)
    async def purge():
        alive_nodes = {node_id for node_id, node_info in DataSource.nodes.items() if node_info['state'] == 'ALIVE'}
        for key in DataSource.node_stats.keys() - alive_nodes:
            DataSource.node_stats.pop(key)
        for key in DataSource.node_physical_stats.keys() - alive_nodes:
            DataSource.node_physical_stats.pop(key)

    @classmethod
    @async_loop_forever(dashboard_consts.ORGANIZE_DATA_INTERVAL_SECONDS)
    async def organize(cls):
        node_workers = {}
        core_worker_stats = {}
        for node_id in list(DataSource.nodes.keys()):
            workers = await cls.get_node_workers(node_id)
            for worker in workers:
                for stats in worker.get('coreWorkerStats', []):
                    worker_id = stats['workerId']
                    core_worker_stats[worker_id] = stats
            node_workers[node_id] = workers
        DataSource.node_workers.reset(node_workers)
        DataSource.core_worker_stats.reset(core_worker_stats)

    @classmethod
    async def get_node_workers(cls, node_id):
        workers = []
        node_physical_stats = DataSource.node_physical_stats.get(node_id, {})
        node_stats = DataSource.node_stats.get(node_id, {})
        pid_to_worker_stats = {}
        pid_to_language = {}
        pid_to_job_id = {}
        pids_on_node = set()
        for core_worker_stats in node_stats.get('coreWorkersStats', []):
            pid = core_worker_stats['pid']
            pids_on_node.add(pid)
            pid_to_worker_stats.setdefault(pid, []).append(core_worker_stats)
            pid_to_language[pid] = core_worker_stats['language']
            pid_to_job_id[pid] = core_worker_stats['jobId']
        for worker in node_physical_stats.get('workers', []):
            worker = dict(worker)
            pid = worker['pid']
            worker['coreWorkerStats'] = pid_to_worker_stats.get(pid, [])
            worker['language'] = pid_to_language.get(pid, dashboard_consts.DEFAULT_LANGUAGE)
            worker['jobId'] = pid_to_job_id.get(pid, dashboard_consts.DEFAULT_JOB_ID)
            await GlobalSignals.worker_info_fetched.send(node_id, worker)
            workers.append(worker)
        return workers

    @classmethod
    async def get_node_info(cls, node_id, get_summary=False):
        node_physical_stats = dict(DataSource.node_physical_stats.get(node_id, {}))
        node_stats = dict(DataSource.node_stats.get(node_id, {}))
        node = DataSource.nodes.get(node_id, {})
        if get_summary:
            node_physical_stats.pop('workers', None)
            node_stats.pop('workersStats', None)
        else:
            node_stats.pop('coreWorkersStats', None)
        store_stats = node_stats.get('storeStats', {})
        used = int(store_stats.get('objectStoreBytesUsed', 0))
        total = int(store_stats.get('objectStoreBytesAvail', 0))
        ray_stats = {'object_store_used_memory': used, 'object_store_available_memory': total - used}
        node_info = node_physical_stats
        node_info['raylet'] = node_stats
        node_info['raylet'].update(ray_stats)
        node_info['status'] = node['stateSnapshot']['state']
        node_info['raylet'].update(node)
        if not get_summary:
            node_info['actors'] = DataSource.node_actors.get(node_id, {})
            node_info['workers'] = DataSource.node_workers.get(node_id, [])
        if get_summary:
            await GlobalSignals.node_summary_fetched.send(node_info)
        else:
            await GlobalSignals.node_info_fetched.send(node_info)
        return node_info

    @classmethod
    async def get_all_node_summary(cls):
        return [await DataOrganizer.get_node_info(node_id, get_summary=True) for node_id in DataSource.nodes.keys()]

    @classmethod
    async def get_all_node_details(cls):
        return [await DataOrganizer.get_node_info(node_id) for node_id in DataSource.nodes.keys()]

    @classmethod
    async def get_all_agent_infos(cls):
        agent_infos = dict()
        for node_id, (http_port, grpc_port) in DataSource.agents.items():
            agent_infos[node_id] = dict(ipAddress=DataSource.node_id_to_ip[node_id], httpPort=int(http_port or -1), grpcPort=int(grpc_port or -1), httpAddress=f'{DataSource.node_id_to_ip[node_id]}:{http_port}')
        return agent_infos

    @classmethod
    async def get_all_actors(cls):
        result = {}
        for index, (actor_id, actor) in enumerate(DataSource.actors.items()):
            result[actor_id] = await cls._get_actor(actor)
            if index % 1000 == 0 and index > 0:
                await asyncio.sleep(0)
        return result

    @staticmethod
    async def _get_actor(actor):
        actor = dict(actor)
        worker_id = actor['address']['workerId']
        core_worker_stats = DataSource.core_worker_stats.get(worker_id, {})
        actor_constructor = core_worker_stats.get('actorTitle', 'Unknown actor constructor')
        actor['actorConstructor'] = actor_constructor
        actor.update(core_worker_stats)
        node_id = actor['address']['rayletId']
        pid = core_worker_stats.get('pid')
        node_physical_stats = DataSource.node_physical_stats.get(node_id, {})
        actor_process_stats = None
        actor_process_gpu_stats = []
        if pid:
            for process_stats in node_physical_stats.get('workers', []):
                if process_stats['pid'] == pid:
                    actor_process_stats = process_stats
                    break
            for gpu_stats in node_physical_stats.get('gpus', []):
                for process in gpu_stats.get('processes') or []:
                    if process['pid'] == pid:
                        actor_process_gpu_stats.append(gpu_stats)
                        break
        actor['gpus'] = actor_process_gpu_stats
        actor['processStats'] = actor_process_stats
        return actor