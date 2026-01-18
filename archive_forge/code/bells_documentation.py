import warnings
from os import getenv

Even more features than `tqdm.auto` (all the bells & whistles):

- `tqdm.auto`
- `tqdm.tqdm.pandas`
- `tqdm.contrib.telegram`
    + uses `${TQDM_TELEGRAM_TOKEN}` and `${TQDM_TELEGRAM_CHAT_ID}`
- `tqdm.contrib.discord`
    + uses `${TQDM_DISCORD_TOKEN}` and `${TQDM_DISCORD_CHANNEL_ID}`
