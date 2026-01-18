import schedule
import time


def daily_task():
    # Code for the daily task goes here
    print("Running daily task...")


schedule.every().day.at("09:00").do(daily_task)

while True:
    schedule.run_pending()
    time.sleep(1)
