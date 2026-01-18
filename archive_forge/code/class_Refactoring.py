import difflib
from pathlib import Path
from typing import Dict, Iterable, Tuple
from parso import split_lines
from jedi.api.exceptions import RefactoringError
from jedi.inference.value.namespace import ImplicitNSName
class Refactoring:

    def __init__(self, inference_state, file_to_node_changes, renames=()):
        self._inference_state = inference_state
        self._renames = renames
        self._file_to_node_changes = file_to_node_changes

    def get_changed_files(self) -> Dict[Path, ChangedFile]:

        def calculate_to_path(p):
            if p is None:
                return p
            p = str(p)
            for from_, to in renames:
                if p.startswith(str(from_)):
                    p = str(to) + p[len(str(from_)):]
            return Path(p)
        renames = self.get_renames()
        return {path: ChangedFile(self._inference_state, from_path=path, to_path=calculate_to_path(path), module_node=next(iter(map_)).get_root_node(), node_to_str_map=map_) for path, map_ in sorted(self._file_to_node_changes.items(), key=lambda x: x[0] or Path(''))}

    def get_renames(self) -> Iterable[Tuple[Path, Path]]:
        """
        Files can be renamed in a refactoring.
        """
        return sorted(self._renames)

    def get_diff(self):
        text = ''
        project_path = self._inference_state.project.path
        for from_, to in self.get_renames():
            text += 'rename from %s\nrename to %s\n' % (_try_relative_to(from_, project_path), _try_relative_to(to, project_path))
        return text + ''.join((f.get_diff() for f in self.get_changed_files().values()))

    def apply(self):
        """
        Applies the whole refactoring to the files, which includes renames.
        """
        for f in self.get_changed_files().values():
            f.apply()
        for old, new in self.get_renames():
            old.rename(new)