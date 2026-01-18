import contextlib
import functools
import fasteners
from oslo_utils import reflection
from oslo_utils import uuidutils
import tenacity
from taskflow import exceptions
from taskflow import logging
from taskflow.persistence.backends import impl_memory
from taskflow.persistence import models
from taskflow import retry
from taskflow import states
from taskflow import task
from taskflow.utils import misc
class _ProviderLocator(object):
    """Helper to start to better decouple the finding logic from storage.

    WIP: part of the larger effort to cleanup/refactor the finding of named
         arguments so that the code can be more unified and easy to
         follow...
    """

    def __init__(self, transient_results, providers_fetcher, result_fetcher):
        self.result_fetcher = result_fetcher
        self.providers_fetcher = providers_fetcher
        self.transient_results = transient_results

    def _try_get_results(self, looking_for, provider, look_into_results=True, find_potentials=False):
        if provider.name is _TRANSIENT_PROVIDER:
            results = self.transient_results
        else:
            try:
                results = self.result_fetcher(provider.name)
            except (exceptions.NotFound, exceptions.DisallowedAccess):
                if not find_potentials:
                    raise
                else:
                    results = {}
        if look_into_results:
            _item_from_single(provider, results, looking_for)
        return results

    def _find(self, looking_for, scope_walker=None, short_circuit=True, find_potentials=False):
        if scope_walker is None:
            scope_walker = []
        default_providers, atom_providers = self.providers_fetcher(looking_for)
        searched_providers = set()
        providers_and_results = []
        if default_providers:
            for p in default_providers:
                searched_providers.add(p)
                try:
                    provider_results = self._try_get_results(looking_for, p, find_potentials=find_potentials, look_into_results=True)
                except exceptions.NotFound:
                    if not find_potentials:
                        raise
                else:
                    providers_and_results.append((p, provider_results))
            if short_circuit:
                return (searched_providers, providers_and_results)
        if not atom_providers:
            return (searched_providers, providers_and_results)
        atom_providers_by_name = dict(((p.name, p) for p in atom_providers))
        for accessible_atom_names in iter(scope_walker):
            maybe_atom_providers = [atom_providers_by_name[atom_name] for atom_name in accessible_atom_names if atom_name in atom_providers_by_name]
            tmp_providers_and_results = []
            if find_potentials:
                for p in maybe_atom_providers:
                    searched_providers.add(p)
                    tmp_providers_and_results.append((p, {}))
            else:
                for p in maybe_atom_providers:
                    searched_providers.add(p)
                    try:
                        provider_results = self._try_get_results(looking_for, p, find_potentials=find_potentials, look_into_results=False)
                    except exceptions.DisallowedAccess as e:
                        if e.state != states.IGNORE:
                            exceptions.raise_with_cause(exceptions.NotFound, 'Expected to be able to find output %r produced by %s but was unable to get at that providers results' % (looking_for, p))
                        else:
                            LOG.trace('Avoiding using the results of %r (from %s) for name %r because it was ignored', p.name, p, looking_for)
                    else:
                        tmp_providers_and_results.append((p, provider_results))
            if tmp_providers_and_results and short_circuit:
                return (searched_providers, tmp_providers_and_results)
            else:
                providers_and_results.extend(tmp_providers_and_results)
        return (searched_providers, providers_and_results)

    def find_potentials(self, looking_for, scope_walker=None):
        """Returns the accessible **potential** providers."""
        _searched_providers, providers_and_results = self._find(looking_for, scope_walker=scope_walker, short_circuit=False, find_potentials=True)
        return set((p for p, _provider_results in providers_and_results))

    def find(self, looking_for, scope_walker=None, short_circuit=True):
        """Returns the accessible providers."""
        return self._find(looking_for, scope_walker=scope_walker, short_circuit=short_circuit, find_potentials=False)