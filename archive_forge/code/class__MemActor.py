from io import StringIO
from typing import Dict, List
import ray
from ray.data.context import DataContext
@ray.remote(num_cpus=0)
class _MemActor:

    def __init__(self):
        self.allocated: Dict[ray.ObjectRef, dict] = {}
        self.deallocated: Dict[ray.ObjectRef, dict] = {}
        self.skip_dealloc: Dict[ray.ObjectRef, str] = {}
        self.peak_mem = 0
        self.cur_mem = 0

    def trace_alloc(self, ref: List[ray.ObjectRef], loc: str):
        ref = ref[0]
        if ref not in self.allocated:
            meta = ray.experimental.get_object_locations([ref])
            size_bytes = meta.get('object_size', 0)
            if not size_bytes:
                size_bytes = -1
                from ray import cloudpickle as pickle
                try:
                    obj = ray.get(ref, timeout=5.0)
                    size_bytes = len(pickle.dumps(obj))
                except Exception:
                    print('[mem_tracing] ERROR getting size')
                    size_bytes = -1
            print(f'[mem_tracing] Allocated {size_bytes} bytes at {loc}: {ref}')
            entry = {'size_bytes': size_bytes, 'loc': loc}
            self.allocated[ref] = entry
            self.cur_mem += size_bytes
            self.peak_mem = max(self.cur_mem, self.peak_mem)

    def trace_dealloc(self, ref: List[ray.ObjectRef], loc: str, freed: bool):
        ref = ref[0]
        size_bytes = self.allocated.get(ref, {}).get('size_bytes', 0)
        if freed:
            print(f'[mem_tracing] Freed {size_bytes} bytes at {loc}: {ref}')
            if ref in self.allocated:
                self.cur_mem -= size_bytes
                self.deallocated[ref] = self.allocated.pop(ref)
                self.deallocated[ref]['dealloc_loc'] = loc
            if ref in self.deallocated:
                pass
            else:
                print(f'[mem_tracing] WARNING: allocation of {ref} was not traced!')
        else:
            print(f'[mem_tracing] Skipped freeing {size_bytes} bytes at {loc}: {ref}')
            self.skip_dealloc[ref] = loc

    def leak_report(self) -> str:
        output = StringIO()
        output.write('[mem_tracing] ===== Leaked objects =====\n')
        for ref in self.allocated:
            size_bytes = self.allocated[ref].get('size_bytes')
            loc = self.allocated[ref].get('loc')
            if ref in self.skip_dealloc:
                dealloc_loc = self.skip_dealloc[ref]
                output.write(f'[mem_tracing] Leaked object, created at {loc}, size {size_bytes}, skipped dealloc at {dealloc_loc}: {ref}\n')
            else:
                output.write(f'[mem_tracing] Leaked object, created at {loc}, size {size_bytes}: {ref}\n')
        output.write('[mem_tracing] ===== End leaked objects =====\n')
        output.write('[mem_tracing] ===== Freed objects =====\n')
        for ref in self.deallocated:
            size_bytes = self.deallocated[ref].get('size_bytes')
            loc = self.deallocated[ref].get('loc')
            dealloc_loc = self.deallocated[ref].get('dealloc_loc')
            output.write(f'[mem_tracing] Freed object from {loc} at {dealloc_loc}, size {size_bytes}: {ref}\n')
        output.write('[mem_tracing] ===== End freed objects =====\n')
        output.write(f'[mem_tracing] Peak size bytes {self.peak_mem}\n')
        output.write(f'[mem_tracing] Current size bytes {self.cur_mem}\n')
        return output.getvalue()