from typing import TYPE_CHECKING, Dict, Optional
class WBURLs:
    _urls_dict: Optional[Dict['URLS', str]]

    def __init__(self) -> None:
        self._urls_dict = None

    def _get_urls(self) -> Dict['URLS', str]:
        return dict(cli_launch='https://wandb.me/launch', doc_run='https://wandb.me/run', doc_require='https://wandb.me/library-require', doc_start_err='https://docs.wandb.ai/guides/track/tracking-faq#initstarterror-error-communicating-with-wandb-process-', doc_artifacts_guide='https://docs.wandb.ai/guides/artifacts', upgrade_server='https://wandb.me/server-upgrade', multiprocess='http://wandb.me/init-multiprocess', wandb_init='https://wandb.me/wandb-init', wandb_server='https://wandb.me/wandb-server', wandb_define_metric='https://wandb.me/define-metric', wandb_core='https://wandb.me/wandb-core')

    def get(self, s: 'URLS') -> str:
        if self._urls_dict is None:
            self._urls_dict = self._get_urls()
        return self._urls_dict[s]